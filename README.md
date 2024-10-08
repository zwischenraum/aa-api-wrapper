# Aleph Alpha API Wrapper

This project is a FastAPI-based wrapper for the Aleph Alpha API, providing endpoints for chat completions, text completions, and embeddings. It acts as a proxy to transform requests and responses to be compatible with Aleph Alpha's API.

## Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**

   Create a `.env` file in the root directory and add your Aleph Alpha API base URL:

   ```
   ALEPH_ALPHA_API_BASE=https://api.aleph-alpha.com
   ```

## Usage

1. **Run the FastAPI server:**

   ```bash
   uvicorn src.main:app --reload
   ```

2. **Available Endpoints:**

   - **Chat Completions:** POST `/v1/chat/completions`
   - **Text Completions:** POST `/v1/completions`
   - **Embeddings:** POST `/v1/embeddings`

3. **Example Request:**

   For text completions, send a POST request to `/v1/completions` with a JSON body:

   ```json
   {
     "prompt": "Once upon a time",
     "max_tokens": 50
   }
   ```

   The wrapper will transform this request to be compatible with Aleph Alpha's API.

## Project Documentation

- **`src/main.py`:** Contains the FastAPI application and endpoint definitions.
- **`proxy_request`:** A utility function to forward requests to the Aleph Alpha API.
- **Transform Functions:** Modify request bodies to match Aleph Alpha's expected format.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
