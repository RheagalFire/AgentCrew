import pytest
import sys
import types
from unittest import mock

# Stub litellm before importing the service
_fake_litellm = types.ModuleType("litellm")

_fake_usage = mock.MagicMock()
_fake_usage.prompt_tokens = 10
_fake_usage.completion_tokens = 5
_fake_usage.prompt_tokens_details = None

_fake_message = mock.MagicMock()
_fake_message.content = "Hello from LiteLLM!"
_fake_message.refusal = None

_fake_choice = mock.MagicMock()
_fake_choice.message = _fake_message
_fake_choice.delta = mock.MagicMock()
_fake_choice.delta.content = "Hello from LiteLLM!"
_fake_choice.delta.tool_calls = None

_fake_response = mock.MagicMock()
_fake_response.choices = [_fake_choice]
_fake_response.usage = _fake_usage
_fake_response.model = "openai/gpt-4o-mini"

_fake_litellm.acompletion = mock.AsyncMock(return_value=_fake_response)
sys.modules["litellm"] = _fake_litellm


@pytest.fixture(autouse=True)
def reset_mocks():
    _fake_litellm.acompletion.reset_mock()
    _fake_litellm.acompletion.return_value = _fake_response
    _fake_message.content = "Hello from LiteLLM!"
    _fake_usage.prompt_tokens = 10
    _fake_usage.completion_tokens = 5
    _fake_usage.prompt_tokens_details = None
    yield


@pytest.fixture
def service():
    from AgentCrew.modules.litellm import LiteLLMService

    svc = LiteLLMService()
    svc.model = "openai/gpt-4o-mini"
    return svc


@pytest.mark.asyncio
async def test_process_message(service):
    async def fake_stream():
        yield _fake_response
        final_chunk = mock.MagicMock()
        final_chunk.choices = []
        final_chunk.usage = _fake_usage
        yield final_chunk

    _fake_litellm.acompletion = mock.AsyncMock(return_value=fake_stream())
    result = await service.process_message("Hello")
    assert "Hello from LiteLLM!" in result


@pytest.mark.asyncio
async def test_process_message_drop_params(service):
    async def fake_stream():
        yield _fake_response

    _fake_litellm.acompletion = mock.AsyncMock(return_value=fake_stream())
    await service.process_message("Hello")
    call_kwargs = _fake_litellm.acompletion.call_args[1]
    assert call_kwargs["drop_params"] is True


@pytest.mark.asyncio
async def test_stream_assistant_response_drop_params(service):
    _fake_litellm.acompletion = mock.AsyncMock(return_value=mock.MagicMock())
    await service.stream_assistant_response([{"role": "user", "content": "Hi"}])
    call_kwargs = _fake_litellm.acompletion.call_args[1]
    assert call_kwargs["drop_params"] is True


@pytest.mark.asyncio
async def test_api_key_forwarded(service):
    service.api_key = "sk-test-123"
    _fake_litellm.acompletion = mock.AsyncMock(return_value=mock.MagicMock())
    await service.stream_assistant_response([{"role": "user", "content": "Hi"}])
    call_kwargs = _fake_litellm.acompletion.call_args[1]
    assert call_kwargs["api_key"] == "sk-test-123"


@pytest.mark.asyncio
async def test_api_key_omitted_when_none(service):
    service.api_key = None
    _fake_litellm.acompletion = mock.AsyncMock(return_value=mock.MagicMock())
    await service.stream_assistant_response([{"role": "user", "content": "Hi"}])
    call_kwargs = _fake_litellm.acompletion.call_args[1]
    assert "api_key" not in call_kwargs


@pytest.mark.asyncio
async def test_api_base_forwarded(service):
    service.api_base = "http://localhost:4000"
    _fake_litellm.acompletion = mock.AsyncMock(return_value=mock.MagicMock())
    await service.stream_assistant_response([{"role": "user", "content": "Hi"}])
    call_kwargs = _fake_litellm.acompletion.call_args[1]
    assert call_kwargs["api_base"] == "http://localhost:4000"


@pytest.mark.asyncio
async def test_validate_spec(service):
    _fake_litellm.acompletion = mock.AsyncMock(return_value=_fake_response)
    result = await service.validate_spec("Validate this")
    assert result == "Hello from LiteLLM!"
    call_kwargs = _fake_litellm.acompletion.call_args[1]
    assert call_kwargs["drop_params"] is True


@pytest.mark.asyncio
async def test_close_noop(service):
    await service.close()


def test_init_defaults(service):
    assert service.model == "openai/gpt-4o-mini"
    assert service._provider_name == "litellm"
    assert service.tools == []
    assert service.tool_handlers == {}
    assert service.system_prompt == "You are a helpful assistant"


def test_set_system_prompt(service):
    service.set_system_prompt("Custom prompt")
    assert service.system_prompt == "Custom prompt"


def test_register_tool(service):
    tool_def = {"type": "function", "function": {"name": "test_tool", "parameters": {}}}
    handler = mock.AsyncMock()
    service.register_tool(tool_def, handler)
    assert "test_tool" in service.tool_handlers
    assert len(service.tools) == 1


def test_clear_tools(service):
    service.tools = [{"name": "test"}]
    service.tool_handlers = {"test": mock.AsyncMock()}
    service.clear_tools()
    assert service.tools == []
    assert service.tool_handlers == {}


def test_model_list_empty_by_default():
    from AgentCrew.modules.litellm.models import LITELLM_MODELS

    assert isinstance(LITELLM_MODELS, list)
    assert len(LITELLM_MODELS) == 0


def test_provider_prefixed_model_string(service):
    assert "/" in service.model


# ---------------------------------------------------------------------------
# Error handling tests (exceptions propagate like other AgentCrew providers)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_auth_error_propagates(service):
    """Auth errors from litellm propagate to the framework."""
    exc = Exception("AuthenticationError: Invalid API key")
    _fake_litellm.acompletion = mock.AsyncMock(side_effect=exc)
    with pytest.raises(Exception, match="Invalid API key"):
        await service.stream_assistant_response([{"role": "user", "content": "Hi"}])


@pytest.mark.asyncio
async def test_rate_limit_propagates(service):
    """Rate limit errors from litellm propagate to the framework."""
    exc = Exception("RateLimitError: 429 Too Many Requests")
    _fake_litellm.acompletion = mock.AsyncMock(side_effect=exc)
    with pytest.raises(Exception, match="429"):
        await service.stream_assistant_response([{"role": "user", "content": "Hi"}])


@pytest.mark.asyncio
async def test_timeout_propagates(service):
    """Timeout errors from litellm propagate to the framework."""
    exc = Exception("Timeout: Request timed out")
    _fake_litellm.acompletion = mock.AsyncMock(side_effect=exc)
    with pytest.raises(Exception, match="timed out"):
        await service.stream_assistant_response([{"role": "user", "content": "Hi"}])


@pytest.mark.asyncio
async def test_not_found_propagates(service):
    """Model not found errors propagate to the framework."""
    exc = Exception("NotFoundError: Model openai/nonexistent not found")
    _fake_litellm.acompletion = mock.AsyncMock(side_effect=exc)
    with pytest.raises(Exception, match="not found"):
        await service.stream_assistant_response([{"role": "user", "content": "Hi"}])


@pytest.mark.asyncio
async def test_validate_spec_empty_content(service):
    """validate_spec returns empty string when content is None."""
    resp = mock.MagicMock()
    resp.choices = [mock.MagicMock()]
    resp.choices[0].message.content = None
    resp.usage = _fake_usage
    _fake_litellm.acompletion = mock.AsyncMock(return_value=resp)
    result = await service.validate_spec("Validate")
    assert result == ""


@pytest.mark.asyncio
async def test_stream_non_streamable_model(service):
    """Non-streamable models return AsyncIterator wrapping choices."""
    from AgentCrew.modules.llm.base import AsyncIterator

    resp = mock.MagicMock()
    resp.choices = [_fake_choice]
    resp.usage = _fake_usage
    _fake_litellm.acompletion = mock.AsyncMock(return_value=resp)

    # Patch _build_stream_params to return non-streamable
    original_build = service._build_stream_params
    def mock_build():
        params, _ = original_build()
        return params, False
    service._build_stream_params = mock_build

    result = await service.stream_assistant_response([{"role": "user", "content": "Hi"}])
    assert isinstance(result, AsyncIterator)


@pytest.mark.asyncio
async def test_process_message_empty_stream(service):
    """process_message returns empty when stream yields no content."""
    async def empty_stream():
        chunk = mock.MagicMock()
        chunk.choices = []
        chunk.usage = _fake_usage
        yield chunk

    _fake_litellm.acompletion = mock.AsyncMock(return_value=empty_stream())
    result = await service.process_message("Hello")
    assert result == ""
