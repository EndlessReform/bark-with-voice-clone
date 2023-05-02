# Bark fine-tuning experiment

> 2023-05-01: So, er, hi everyone from the Serp.ai Discord! Didn't think anyone would actually find this repo until I was finished. Not sure how much help I'd be, but I'd love to contribute to the main fine-tuning effort!

> **warning**
>
> I'm a junior web dev with a grand total of four months of AI tutorials, so I could be totally "Bark"-ing up the wrong tree! Please don't hesitate to give suggestions, contribute, or correct me, that's what open source is for!

This repo attempts to enable converting ground-truth audio to Bark semantic tokens (or their input embeddings). If successful, this will add the missing piece to Serp.ai's voice cloning fork, which solved coarse and fine token conversion, and enable full fine tuning - or at least get some of the way there. **My eventual goal is to merge this fork back into the main Serp.ai voice cloning fork**, if I ever get that far.

## Why can't Bark be fine-tuned (yet)?

Under the hood, Bark is essentially the AudioLM model (see [paper](https://arxiv.org/abs/2209.03143), [public GitHub replication](https://github.com/lucidrains/audiolm-pytorch)) + text conditioning. It's three GPTs stacked on top of each other. In AudioLM, just like GPT-3 generates text tokens from a prompt of text tokens, the first GPT takes a prompt of **semantic** tokens, which encode the _content_ of new audio and a bit of the speaker identity (that's `text_to_semantic` in Bark), and generates the "next tokens". Bark adds to this by adding a learned embedding of the text you want to generate. The second and third GPTs, the fine and coarse or `semantic_to_waveform` in Bark, in both Bark and AudioLM handle the **acoustic** tokens, which encode the finer details of the audio.

![](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEiX0MyO_IXk730mCbbJX7LXxBRIxJk2K41Y4leuEk4WQRjz0kgIp9CGHFwLePaKt3qEcCK8fvhAxjJ7J_sXH05q7xnsMbjZZFDDLPIlVyaKr3yYo77oT2KBqe9gw4MFuUZnUfxFprP67ExPzr2RNxduB0SruUGjJXghihHSoxvMtlG3YNtHHesZOJzY/s960/image2.png)

So how do you turn real audio into token prompts for these models? Mercifully, the acoustic tokens are a predefined open-source format: lower and upper layers of Facebook's [Encodec](https://github.com/facebookresearch/encodec) neural compressed encoding for audio. Serp.ai's voice cloning fork successfully converts coarse and fine token prompts this way. Unfortunately, the audio to semantic token conversion requires Bark's proprietary model, which is only used during training and not at inference. Suno has repeatedly refused to open-source this model despite many community requests including from Serp.ai, in order to ~~make money by using an unconfigurable Bark as a giant advertisement for their future proprietary platform, which guards cloning behind a paid API~~ "prevent online harms and misinformation". Instead, Suno gives out their own predefined prompts. This approach is quite similar to how the Tortoise TTS developer "de-weaponized" it for the first year of its existence: see Sherman Chann's blog post [Why can't Tortoise be fine-tuned?](https://152334h.github.io/blog/tortoise-fine-tuning/) for a writeup.

Serp.ai's voice cloning fork deals with this limitation by generating semantic tokens prompted only by text, but supplying the fine and coarse prompts from the ground-truth audio. Serp's approach gets pretty far; fine and coarse are enough to get major details like speaker gender and tone of voice pretty close. However, sadly this isn't enough to nail speaker identity. Check out the `notebooks/ablation.ipynb` notebook for an informal demonstration of how much difference semantic and acoustic prompts make to the output.

## Reverse engineering the semantic token codebook

Sherman Chann's blog post on Tortoise goes on to suggest "baseless speculation" on how to reverse-engineer the Tortoise codebook. By definition, the model outputs are the audio from the new semantic tokens, and mercifully, the length specified by the 50hz semantic tokens is the length of the audio. So we can generate a large, diverse dataset of voice lines and save the semantic tokens for them, then train a small model to map generated audio to source tokens. Chann never ended up having to do this, since the Tortoise author foolishly left the original semantic token encoder in a not-actually-deleted HuggingFace branch. Sadly, the Bark community isn't so lucky; we'll have to do it the hard way.

The `notebooks/create_dataset` is a naive attempt to generate a dataset of synthetic audio to semantic tokens, in [Fairseq's](https://github.com/facebookresearch/fairseq) dataset format, so we can feed our generated audio easily into Fairseq's HuBERT implementation and get the sixth-layer embeddings. The key thing here is to generate as large and diverse a dataset as possible, but for prototyping purposes, I'm solely doing this for English using voice prompts from [Mozilla CommonVoice](https://commonvoice.mozilla.org/en/datasets) (NOT the actual audio). (As a side note, I would really appreciate someone getting the `validated.tsv` voice lines from other languages in CommonVoice, like Hindi; I don't want to download all that audio just to get the tsv and not use the audio at all).

The original AudioLM paper creates the audio to semantic token mapping as follows:
- Take an encoder transformer BERT-like model that encodes audio to embeddings (for tasks like speaker recognition). AudioLM and Google use the closed-source wav2vec-BERT, but the open-source AudioLM repo uses [HuBERT](https://huggingface.co/docs/transformers/model_doc/hubert).
- Run a bunch of source audio through HuBERT and take the embeddings from the sixth layer. HuBERT runs at 50 embeddings / second of audio.
- Run k-means clusters on the embeddings to essentially produce k "groups" of kinds of input audio. For example, AudioLM uses ~500, and in a [GitHub statement](https://github.com/lucidrains/audiolm-pytorch/discussions/170), the Bark devs say they use a similar approach but with 10k groups. In what I am sure is a complete coincidence, Bark semantic tokens are 49.9hz, roughly the same as HuBERT's 50hz.
- When adding new audio, run k-means to find out "what group" the new audio is in.

So can't we just do this semantic token codebook generation ourselves? No; as Chann points out, there's no guarantee that our own training process will generate the same groups. Instead, there are two different rather naive approaches I'm going to try:

- Divide the generations by semantic token, find the means of the HuBERT embeddings of the generated audio that correspond to that token's position in the semantic prompt, and use them as starting centroids for a small K-means run. Then run normal K-means inference.
- Similar to [Mini-GPT-4](https://arxiv.org/abs/2304.10592), simply train a linear projection from embeddings from frozen HuBERT to Bark's input embeddings for the semantic tokens, token by token. Leave Bark and HuBERT frozen. In my uneducated opinion this is less stupid than it sounds; since a simple linear projection was enough to map high-dimensional image embeddings to text input embeddings, surely mapping two kinds of 50hz audio representations can't be _that_ hard? (Knock on wood). Also, input embeddings aren't reliant on the tokens around them. 

Other stuff that probably needs to be done later:
- Add batch inference mode for Bark, to speed up dataset generation and enable use cases like mass audiobook conversion
- Write an eval harness, so we can gauge performance better than training objective loss or "playing it by ear"

-------------------------------------------------------------------
# Original README.md

<a href="http://www.repostatus.org/#active"><img src="http://www.repostatus.org/badges/latest/active.svg" /></a>
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/OnusFM.svg?style=social&label=@OnusFM)](https://twitter.com/OnusFM)
[![](https://dcbadge.vercel.app/api/server/J2B2vsjKuE?compact=true&style=flat&)](https://discord.gg/J2B2vsjKuE)


[Examples](https://suno-ai.notion.site/Bark-Examples-5edae8b02a604b54a42244ba45ebc2e2) | [Model Card](./model-card.md) | [Playground Waitlist](https://3os84zs17th.typeform.com/suno-studio)

Bark is a transformer-based text-to-audio model created by [Suno](https://suno.ai). Bark can generate highly realistic, multilingual speech as well as other audio - including music, background noise and simple sound effects. The model can also produce nonverbal communications like laughing, sighing and crying. To support the research community, we are providing access to pretrained model checkpoints ready for inference.

<p align="center">
<img src="https://user-images.githubusercontent.com/5068315/230698495-cbb1ced9-c911-4c9a-941d-a1a4a1286ac6.png" width="500"></img>
</p>

## 🔊 Demos

[![Open in Spaces](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/spaces/suno/bark)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eJfA2XUa-mXwdMy7DoYKVYHI1iTd9Vkt?usp=sharing)

## 🤖 Usage

```python
from bark import SAMPLE_RATE, generate_audio, preload_models
from IPython.display import Audio

# download and load all models
preload_models()

# generate audio from text
text_prompt = """
     Hello, my name is Suno. And, uh — and I like pizza. [laughs] 
     But I also have other interests such as playing tic tac toe.
"""
audio_array = generate_audio(text_prompt)

# play text in notebook
Audio(audio_array, rate=SAMPLE_RATE)
```

[pizza.webm](https://user-images.githubusercontent.com/5068315/230490503-417e688d-5115-4eee-9550-b46a2b465ee3.webm)


To save `audio_array` as a WAV file:

```python
from scipy.io.wavfile import write as write_wav

write_wav("/path/to/audio.wav", SAMPLE_RATE, audio_array)
```

### 🌎 Foreign Language

Bark supports various languages out-of-the-box and automatically determines language from input text. When prompted with code-switched text, Bark will attempt to employ the native accent for the respective languages. English quality is best for the time being, and we expect other languages to further improve with scaling. 

```python
text_prompt = """
    Buenos días Miguel. Tu colega piensa que tu alemán es extremadamente malo. 
    But I suppose your english isn't terrible.
"""
audio_array = generate_audio(text_prompt)
```

[miguel.webm](https://user-images.githubusercontent.com/5068315/230684752-10baadfe-1e7c-46a2-8323-43282aef2c8c.webm)

### 🎶 Music

Bark can generate all types of audio, and, in principle, doesn't see a difference between speech and music. Sometimes Bark chooses to generate text as music, but you can help it out by adding music notes around your lyrics.

```python
text_prompt = """
    ♪ In the jungle, the mighty jungle, the lion barks tonight ♪
"""
audio_array = generate_audio(text_prompt)
```

[lion.webm](https://user-images.githubusercontent.com/5068315/230684766-97f5ea23-ad99-473c-924b-66b6fab24289.webm)

### 🎤 Voice Presets and Voice/Audio Cloning

Bark has the capability to fully clone voices - including tone, pitch, emotion and prosody. The model also attempts to preserve music, ambient noise, etc. from input audio. However, to mitigate misuse of this technology, we limit the audio history prompts to a limited set of Suno-provided, fully synthetic options to choose from for each language. Specify following the pattern: `{lang_code}_speaker_{0-9}`.

```python
text_prompt = """
    I have a silky smooth voice, and today I will tell you about 
    the exercise regimen of the common sloth.
"""
audio_array = generate_audio(text_prompt, history_prompt="en_speaker_1")
```


[sloth.webm](https://user-images.githubusercontent.com/5068315/230684883-a344c619-a560-4ff5-8b99-b4463a34487b.webm)

*Note: since Bark recognizes languages automatically from input text, it is possible to use for example a german history prompt with english text. This usually leads to english audio with a german accent.*

### 👥 Speaker Prompts

You can provide certain speaker prompts such as NARRATOR, MAN, WOMAN, etc. Please note that these are not always respected, especially if a conflicting audio history prompt is given.

```python
text_prompt = """
    WOMAN: I would like an oatmilk latte please.
    MAN: Wow, that's expensive!
"""
audio_array = generate_audio(text_prompt)
```

[latte.webm](https://user-images.githubusercontent.com/5068315/230684864-12d101a1-a726-471d-9d56-d18b108efcb8.webm)


## 💻 Installation

```
pip install git+https://github.com/suno-ai/bark.git
```

or

```
git clone https://github.com/suno-ai/bark
cd bark && pip install . 
```

## 🛠️ Hardware and Inference Speed

Bark has been tested and works on both CPU and GPU (`pytorch 2.0+`, CUDA 11.7 and CUDA 12.0).
Running Bark requires running >100M parameter transformer models.
On modern GPUs and PyTorch nightly, Bark can generate audio in roughly realtime. On older GPUs, default colab, or CPU, inference time might be 10-100x slower. 

If you don't have new hardware available or if you want to play with bigger versions of our models, you can also sign up for early access to our model playground [here](https://3os84zs17th.typeform.com/suno-studio).

## ⚙️ Details

Similar to [Vall-E](https://arxiv.org/abs/2301.02111) and some other amazing work in the field, Bark uses GPT-style 
models to generate audio from scratch. Different from Vall-E, the initial text prompt is embedded into high-level semantic tokens without the use of phonemes. It can therefore generalize to arbitrary instructions beyond speech that occur in the training data, such as music lyrics, sound effects or other non-speech sounds. A subsequent second model is used to convert the generated semantic tokens into audio codec tokens to generate the full waveform. To enable the community to use Bark via public code we used the fantastic 
[EnCodec codec](https://github.com/facebookresearch/encodec) from Facebook to act as an audio representation.

Below is a list of some known non-speech sounds, but we are finding more every day. Please let us know if you find patterns that work particularly well on [Discord](https://discord.gg/J2B2vsjKuE)!

- `[laughter]`
- `[laughs]`
- `[sighs]`
- `[music]`
- `[gasps]`
- `[clears throat]`
- `—` or `...` for hesitations
- `♪` for song lyrics
- capitalization for emphasis of a word
- `MAN/WOMAN:` for bias towards speaker

**Supported Languages**

| Language | Status |
| --- | --- |
| English (en) | ✅ |
| German (de) | ✅ |
| Spanish (es) | ✅ |
| French (fr) | ✅ |
| Hindi (hi) | ✅ |
| Italian (it) | ✅ |
| Japanese (ja) | ✅ |
| Korean (ko) | ✅ |
| Polish (pl) | ✅ |
| Portuguese (pt) | ✅ |
| Russian (ru) | ✅ |
| Turkish (tr) | ✅ |
| Chinese, simplified (zh) | ✅ |
| Arabic  | Coming soon! |
| Bengali | Coming soon! |
| Telugu | Coming soon! |

## 🙏 Appreciation

- [nanoGPT](https://github.com/karpathy/nanoGPT) for a dead-simple and blazing fast implementation of GPT-style models
- [EnCodec](https://github.com/facebookresearch/encodec) for a state-of-the-art implementation of a fantastic audio codec
- [AudioLM](https://github.com/lucidrains/audiolm-pytorch) for very related training and inference code
- [Vall-E](https://arxiv.org/abs/2301.02111), [AudioLM](https://arxiv.org/abs/2209.03143) and many other ground-breaking papers that enabled the development of Bark

## © License

Bark is licensed under a non-commercial license: CC-BY 4.0 NC. The Suno models themselves may be used commercially. However, this version of Bark uses `EnCodec` as a neural codec backend, which is licensed under a [non-commercial license](https://github.com/facebookresearch/encodec/blob/main/LICENSE).

Please contact us at `bark@suno.ai` if you need access to a larger version of the model and/or a version of the model you can use commercially.  

## 📱 Community

- [Twitter](https://twitter.com/OnusFM)
- [Discord](https://discord.gg/J2B2vsjKuE)

## 🎧 Suno Studio (Early Access)

We’re developing a playground for our models, including Bark. 

If you are interested, you can sign up for early access [here](https://3os84zs17th.typeform.com/suno-studio).

## FAQ

#### How do I specify where models are downloaded and cached?

Use the `XDG_CACHE_HOME` env variable to override where models are downloaded and cached (otherwise defaults to a subdirectory of `~/.cache`).

#### Bark's generations sometimes differ from my prompts. What's happening?

Bark is a GPT-style model. As such, it may take some creative liberties in its generations, resulting in higher-variance model outputs than traditional text-to-speech approaches.
