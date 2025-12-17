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

## Documentation
Common patterns and usage examples are documented in [project wiki](https://github.com/izxxr/kevin/wiki). There is no API reference
available at the moment however all functions and classes are well documented in the code. Quickstart code is shown below.

## Example Usage
Below is the example quickstart code.

**This code has various dependencies before it can be run. Please refer to wiki for detailed quickstart guide and explanation of the code.**

```python
import kevin
import webbrowser  # for looking up on Google

inference = kevin.inference.HuggingFaceInferenceBackend(
  "Qwen/Qwen3-4B-Instruct-2507",  # recommended model for most use cases
  token="<hugging-face-api-key>"  # put hugging face API key here
)
stt = kevin.stt.FasterWhisperSTT("tiny.en")
tts = kevin.tts.PiperTTS("<path-to-voice>")  # put path to downloaded Piper voice here
hotword_detector = kevin.hotwords.PorcupineHotwordDetector(
  access_key="<porcupine-access-key>",  # put porcupine access key here
  keywords=["hey siri"]  # default wake word provided by porcupine, use keyword_paths for custom keywords
)

assistant = kevin.Kevin(inference=inference, stt=stt, tts=tts, hotword_detector=hotword_detector)

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
```

## Extending for Other Use Cases
While Kevin was primarily intended and tailored for my own personal use case, it is purely written with generality and flexibility in mind
to adapt maximally to most other use cases.

Although the built-in support is currently limited to specific providers such as FasterWhisper for STT and Piper for TTS etc., abstract
interfaces are available (and well documented) to integrate any other third party provider. Please refer to wiki for details and examples
of extending various abstract classes.

Feel free to suggest other providers that should be added through issues or even contributing an implementation. See [Contributing](#contributing) section.

## Contributing
Kevin is under active development and there may be bugs at this stage. While I'm actively working on this project, mostly for my own use case, I really
appreciate any contributions whether through issues or pull requests to make the library better.

For code contributions, please make sure to include typehints, remain consistent with existing code style, and follow PEP-8 guidelines.
