# Aleph Alpha API Wrapper

This project is a FastAPI-based wrapper for the Aleph Alpha API, providing endpoints for chat completions, text completions, and embeddings. It acts as a proxy to transform requests and responses to be compatible with Aleph Alpha's API.

## Installation

1. **Clone the repository:**

   ```bash
   git clone git@github.com:zwischenraum/aa-api-wrapper.git
   cd aa-api-wrapper
   ```

2. **Install Poetry:**

   If you haven't installed Poetry yet, follow the [official installation guide](https://python-poetry.org/docs/#installation).

3. **Install dependencies:**

   ```bash
   poetry install
   ```

4. **Set up environment variables:**

   Create a `.env` file in the root directory and add your Aleph Alpha API base URL:

   ```
   ALEPH_ALPHA_API_BASE=https://api.aleph-alpha.com
   ```

   and if you want to use semantic embeddings

   ```
   USE_SEMANTIC_EMBEDDINGS=true
   ```

## Usage

2. **Run the FastAPI server:**

   ```bash
   poetry run start
   ```

3. **Available Endpoints:**

   - **Chat Completions:** POST `/v1/chat/completions`
   - **Text Completions:** POST `/v1/completions`
   - **Embeddings:** POST `/v1/embeddings`

4. **Example Request:**

   For text completions, send a POST request to `/v1/completions` with a JSON body:

   ```json
   {
     "prompt": "Once upon a time",
     "max_tokens": 50
   }
   ```

   The wrapper will transform this request to be compatible with Aleph Alpha's API.
