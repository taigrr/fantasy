# Google Provider

This document describes how to get an API keys for Google Gemini and Vertex.

## Build tag

The Gemini/Vertex client (`google.golang.org/genai` plus its gRPC and cloud
auth dependency tree, roughly 14 MB of binary) is compiled in only when the
`fantasy_google` build tag is set:

```sh
go build -tags fantasy_google ./...
```

Without the tag, `google.New` and the option constructors still compile and
`google.Enabled` is `false`; `LanguageModel` returns `google.ErrNotCompiled`.
The same tag gates Vertex AI support in the Anthropic provider
(`anthropic.VertexEnabled`, `anthropic.ErrVertexNotCompiled`).

## Gemini

Simply navigate to [this page](https://aistudio.google.com/apikey) in the
Google AI Studio and create a new API key.

## Vertex

### Install `gcloud`

Install the `gcloud` command line tool. Install via Homebrew, Nix, or download
it from [here](https://cloud.google.com/sdk/docs/install).

```bash
# Homebrew
brew install --cask google-cloud-sdk

# Nix
nix-env -iA nixpkgs.google-cloud-sdk
```

### Authenticate

Then authenticate with your Google account:

```bash
gcloud auth login
```

### Create And Setup Project

Navigate here to create a new project if you haven't already:
https://console.cloud.google.com/projectcreate

Alternatively, you can create a new project via the command line:

```bash
gcloud projects create {YOUR_PROJECT_ID} --name="{YOUR_PROJECT_NAME}"
```

Set the project on your machine:

```bash
gcloud config set project {YOUR_PROJECT_ID}
```

Enable the Vertex AI API:

```bash
gcloud services enable aiplatform.googleapis.com
```

### Setup Env

Finally, you need to run this command to ensure that libraries will be able to
find your credentials.

```bash
gcloud auth application-default login
```
