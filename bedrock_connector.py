#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------
# SOAR Bedrock LLM Connector Python file
# -----------------------------------------

# Phantom App imports
import phantom.app as phantom
from phantom.action_result import ActionResult
from phantom.base_connector import BaseConnector

import requests
import json
import traceback
from typing import Any, Dict, Optional, Tuple
import boto3
from botocore.exceptions import BotoCoreError, ClientError

from bs4 import BeautifulSoup

class RetVal(tuple):

    def __new__(cls, val1, val2=None):
        return tuple.__new__(RetVal, (val1, val2))

class BedrockConnector(BaseConnector):
    """Splunk SOAR connector for invoking Amazon Bedrock LLMs."""
        
    def __init__(self):        
        #super(BedrockConnector, self).__init__()
        super().__init__()        
        self._state = None                        
        
    def initialize(self):
        self._state = self.load_state()
        return phantom.APP_SUCCESS

    def finalize(self):
        self.save_state(self._state or {})
        return phantom.APP_SUCCESS
    
    def _get_required_config(self) -> Tuple[str, str, str, str, str]:
        config = self.get_config()
        access_key = config.get("access_key")
        secret_key = config.get("secret_key")
        region = config.get("region")
        role_arn = config.get("role_arn")
        model_id = config.get("model_id")

        missing = [
            name
            for name, value in (
                ("access_key", access_key),
                ("secret_key", secret_key),
                ("region", region),
                ("role_arn", role_arn),
                ("model_id", model_id),
            )
            if not value
        ]
        if missing:
            raise ValueError("Missing required asset configuration: {}".format(", ".join(missing)))

        return access_key, secret_key, region, role_arn, model_id
            
    def _handle_test_connectivity(self, param):
        self.save_progress("Validating asset configuration")
        action_result = self.add_action_result(ActionResult(dict(param)))
        try:
            _, _, region, _, model_id = self._get_required_config()
            self.save_progress("Assuming configured IAM role")
            session = self._assume_role_session()
            self.save_progress("Creating Bedrock Runtime client in {}".format(region))
            session.client("bedrock-runtime")
            action_result.add_data({"region": region, "model_id": model_id})
            self.save_progress("Test connectivity passed")
            return action_result.set_status(phantom.APP_SUCCESS, "Connectivity test succeeded")
        except Exception as exc:
            self.debug_print(traceback.format_exc())
            return action_result.set_status(phantom.APP_ERROR, "Connectivity test failed: {}".format(exc))

    def _handle_generate_text(self, param):                
        action_result = self.add_action_result(ActionResult(dict(param)))
        try:
            _, _, _, _, default_model_id = self._get_required_config()
            prompt = param.get("prompt")
            if not prompt:
                return action_result.set_status(phantom.APP_ERROR, "Parameter 'prompt' is required")

            model_id = param.get("model_id") or default_model_id
            max_tokens = int(param.get("max_tokens", 1024))
            temperature = float(param.get("temperature", 0.3))
            system_prompt = param.get("system_prompt")

            if max_tokens <= 0:
                return action_result.set_status(phantom.APP_ERROR, "max_tokens must be greater than 0")
            if temperature < 0 or temperature > 1:
                return action_result.set_status(phantom.APP_ERROR, "temperature must be between 0 and 1")

            self.save_progress("Invoking Bedrock model: {}".format(model_id))
            result = self._invoke_bedrock(prompt, model_id, max_tokens, temperature, system_prompt)
            action_result.add_data(result)
            action_result.update_summary({
                "model_id": model_id,
                "response_length": len(result.get("response_text", "")),
            })
            return action_result.set_status(phantom.APP_SUCCESS, "Generated text successfully")
        except (ClientError, BotoCoreError, ValueError, KeyError, json.JSONDecodeError) as exc:
            self.debug_print(traceback.format_exc())
            return action_result.set_status(phantom.APP_ERROR, "Failed to generate text: {}".format(exc))
        except Exception as exc:
            self.debug_print(traceback.format_exc())
            return action_result.set_status(phantom.APP_ERROR, "Unexpected error: {}".format(exc))
        
    def handle_action(self, param):        
        action_id = self.get_action_identifier()
        if action_id == "test_connectivity":
            return self._handle_test_connectivity(param)
        if action_id == "generate_text":
            return self._handle_generate_text(param)
        return phantom.APP_ERROR
                
    def _assume_role_session(self):
        access_key, secret_key, region, role_arn, _ = self._get_required_config()
        base_session = boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )
        sts_client = base_session.client("sts")
        sts_response = sts_client.assume_role(
            RoleArn=role_arn,
            RoleSessionName="splunk-soar-bedrock-llm",
        )
        credentials = sts_response["Credentials"]
        return boto3.Session(
            aws_access_key_id=credentials["AccessKeyId"],
            aws_secret_access_key=credentials["SecretAccessKey"],
            aws_session_token=credentials["SessionToken"],
            region_name=region,
        )
        
    def _build_anthropic_body(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ],
        }
        if system_prompt:
            body["system"] = system_prompt
        return body
    
    def _build_nova_body(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"text": system_prompt}]})
        messages.append({"role": "user", "content": [{"text": prompt}]})
        return {
            "messages": messages,
            "inferenceConfig": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
            },
        }
    
    def _build_body_for_model(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        if model_id.startswith("anthropic."):
            return self._build_anthropic_body(prompt, max_tokens, temperature, system_prompt)
        if model_id.startswith("amazon.nova"):
            return self._build_nova_body(prompt, max_tokens, temperature, system_prompt)

        # Generic fallback for Anthropic-compatible chat payloads. Many Bedrock chat examples use
        # provider-specific JSON schemas. For non-Anthropic/Nova models, update this method with
        # the exact request schema required by the selected model provider.
        return self._build_anthropic_body(prompt, max_tokens, temperature, system_prompt)
    
    def _extract_text_from_response(self, payload: Dict[str, Any]) -> str:
        # Anthropic Claude Messages API format.
        content = payload.get("content")
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif isinstance(item, dict) and "text" in item:
                    text_parts.append(item.get("text", ""))
            if text_parts:
                return "\n".join(text_parts).strip()

        # Amazon Nova response format.
        output = payload.get("output")
        if isinstance(output, dict):
            message = output.get("message", {})
            message_content = message.get("content")
            if isinstance(message_content, list):
                text_parts = [part.get("text", "") for part in message_content if isinstance(part, dict)]
                if text_parts:
                    return "\n".join(text_parts).strip()

        # Other common model response keys.
        for key in ("generation", "outputText", "completion", "text"):
            value = payload.get(key)
            if isinstance(value, str):
                return value.strip()

        return json.dumps(payload, indent=2, sort_keys=True)
    
    def _invoke_bedrock(
        self,
        prompt: str,
        model_id: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        session = self._assume_role_session()
        client = session.client("bedrock-runtime")
        body = self._build_body_for_model(model_id, prompt, max_tokens, temperature, system_prompt)

        response = client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
        payload = json.loads(response["body"].read())
        response_text = self._extract_text_from_response(payload)
        return {
            "model_id": model_id,
            "response_text": response_text,
            "raw_response": payload,
        }

if __name__ == '__main__':    
    import sys
    import pudb

    pudb.set_trace()
    connector = BedrockConnector()
    connector.print_progress_message = True
    result = connector._handle_action(json.loads(sys.stdin.read()))
    sys.exit(0 if result == phantom.APP_SUCCESS else 1)