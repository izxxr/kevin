# KEVIN
> Acronym for **K**nowledgeable **E**ngine for **V**irtual and **I**ntelligent **N**avigation

> Under active development! There may be bugs. Please [open an issue](https://github.com/izxxr/kevin/issues) if
> you encounter one.

`kevin` is a Python library for building simple virtual assistants.

**Features:**

- Built-in support for [FasterWhisper](https://github.com/SYSTRAN/faster-whisper) (STT), [Porcupine](https://github.com/Picovoice/porcupine) (hotword detection),
  [Piper](https://github.com/OHF-Voice/piper1-gpl) (TTS), [HF `InferenceClient`](https://huggingface.co/docs/huggingface_hub/en/package_reference/inference_client) (LLM inference)
- Flexible interface to use other providers for STT, TTS, hotword detection, and LLM inference
- Pydantic based declarative tools definition interface (native support for [function calling](https://huggingface.co/docs/hugs/main/en/guides/function-calling))
- Sensible pre-defined defaults and system prompts yet fully customizable
- Plugins support for modularized tools definitions
- Type annotated and well documented API

## Installation
Kevin can be installed using pip. It is not currently available on PyPi and has to be installed through Git:

```sh
pip install "kevin[default] @ git+https://github.com/izxxr/kevin.git"
```

We are including `default` dependencies that installs the libraries required for using
the built-in providers including:

- `faster-whisper` and `SpeechRecognition` for speech recognition
- `piper-tts` for text to speech
- `huggingface_hub` for LLM response generation
- `pvporcupine` for wake word detection.

## Quickstart

### Hugging Face API Key
Kevin provides built-in support for [Hugging Face Inference Client](https://huggingface.co/docs/huggingface_hub/en/package_reference/inference_client) which allows accessing models and generating responses from LLMs.

In order to use the inference client, API key is required. Create an account on
[Hugging Face](https://huggingface.co) and navigate to https://huggingface.co/settings/tokens/new?tokenType=fineGrained

Tick **"Make calls to Inference Providers"** checkbox and create the token. Copy the access token and store
it somewhere safe.

### Usage
Below is the example quickstart code with basic functionality of Google lookup.

```python
import kevin
import webbrowser  # for looking up on Google

inference = kevin.inference.HuggingFaceInferenceBackend(
  "Qwen/Qwen3-4B-Instruct-2507",  # recommended model for most use cases
  token="<hugging-face-access-token>"  # put generated hugging face access token here
)

assistant = kevin.Kevin(inference=inference)

# Tool is an action that assistant can perform. The class docstring is the description of
# tool that assistant uses to understand when to call the tool. 'topic' is a string parameter
# for this tool that will be extracted from user prompt by assistant.
# For example, if I say "What are volcanoes?" This tool will be called with topic="volcanoes"
@assistant.tool
class LookupGoogle(kevin.tools.Function):
    """Lookup a term, person, place, or any applicable concept on Google."""

    topic: str

    def callback(self, ast):
        webbrowser.open(f"https://google.com/search?q={self.topic}")

assistant.start()
```

Run the code and try prompt such as "what are volcanoes?" or "look up benefits of apples."

### Inference Backend and LLMs
`HuggingFaceInferenceBackend` interacts with the hugging face inference API for generating LLM responses.

You may use any LLM of your choice, but **it must support [tools and function calling](https://huggingface.co/docs/hugs/main/en/guides/function-calling)**. [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) is extremely reliable for most tasks and [`SmolLM3-3B`](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) is good for smaller tasks or testing purposes.

### Speech Recognition and Speaking Assistants
The example shown above is a text mode based assistant. For integrating speech recognition for prompts and
text to speech for assistant's responses, see [this wiki page](https://github.com/izxxr/kevin/wiki/Setting-up-STT-TTS).

## Documentation
There is no API reference available at the moment however all functions and classes are well documented in
the code.

## Extending for Other Use Cases
While Kevin was primarily intended and tailored for my own personal use case, it is purely written with generality and flexibility in mind
to adapt maximally to most other use cases.

Although the built-in support is currently limited to specific providers such as FasterWhisper for STT and Piper for TTS etc., abstract
interfaces are available (and well documented) to integrate any other third party provider.

Feel free to suggest other providers that should be added through issues or even contributing an implementation. See [Contributing](#contributing) section.

## Contributing
Kevin is under active development and there may be bugs at this stage. While I'm actively working on this project, mostly for my own use case, I really
appreciate any contributions whether through issues or pull requests to make the library better.

For code contributions, please make sure to include typehints, remain consistent with existing code style, and follow PEP-8 guidelines.
