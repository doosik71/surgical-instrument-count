# Surgical Instrument Count

## Installation

- Huggingface에서 제공하는 Transformers 패키지를 설치한다.

```bash
uv pip install --no-cache git+https://github.com/huggingface/transformers.git
```

- CLIP의 단어 데이터셋을 다운로드 및 설치한다.

```bash
mkdir .venv/lib/python3.12/site-packages/assets
wget https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz
mv bpe_simple_vocab_16e6.txt.gz .venv/lib/python3.12/site-packages/assets/
```

- Qt6 라이브러리를 설치한다.

```bash
sudo apt install qt6-base-dev libqt6gui6 libqt6widgets6 libxcb-cursor0
```

## Login

- Meta의 SAM3 모델을 Huggingface의 허용된 계정만 접속할 수 있다.
- <https://huggingface.co/facebook/sam3>에서 접근권한을 요청한 후, 터미널에서 Huggingface에 로그인한다.

```bash
huggingface-cli login
```
