import os
import numpy as np
import logging
import jwt
import io
import cv2
import base64
import asyncio
from PyPDF2 import PdfReader, PdfWriter
from pymongo import MongoClient
from pdf2image import convert_from_path, convert_from_bytes
from marshmallow import Schema, fields
from io import BytesIO
from functools import wraps
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from datetime import datetime, timedelta
from Crypto.Cipher import AES
from collections import OrderedDict
import easyocr
import torch
import urllib.parse
import requests


MAX_PAGES = 4
MAX_TIMEOUT = 100.0  # seconds
INITIAL_DPI = 150
MAX_DPI_RANGE = 3
DPI_TO_SKIP = 50
PYTESSERACT_CONFIG = "--oem 1 --psm 12 -l eng"
CUSTOM_MESSAGES = ["MESSAGE"] * 20
CUSTOM_REASONS = ["REASON"] * 20
NEED_TO_SENT_EMAIL = [False] * 20
INFERENCE_API_URL = "http://swayamai.jio.com:5001/inference/_1"

# Initialize Flask app
bills = Flask(__name__)

# Configure logging to log only API hits, errors, and relevant events
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()

# Set a timeout value (in seconds)
TIMEOUT_DURATION = 3600


class Config:
    MONGO_USER = urllib.parse.quote_plus(os.getenv("MONGO_USER", ""))
    MONGO_PASS = urllib.parse.quote_plus(os.getenv("MONGO_PASS", ""))
    MONGO_HOST = os.getenv("MONGO_HOST", "localhost")
    MONGO_PORT = os.getenv("MONGO_PORT", "27017")
    MONGO_AUTH_DB = os.getenv("MONGO_AUTH_DB", "admin")

    if not MONGO_USER or not MONGO_PASS:
        raise ValueError(
            "MongoDB credentials are required but not set in the environment variables."
        )

    MONGO_URI = (
        f"mongodb://{MONGO_USER}:{MONGO_PASS}@{MONGO_HOST}:{MONGO_PORT}/"
        f"?authSource={MONGO_AUTH_DB}&authMechanism=DEFAULT"
    )
    logging.info("Mongo connection URI with credentials configured.")

    AES_KEY = os.getenv("AES_KEY")
    UPLOADS_DIR = os.getenv("UPLOADS_DIR", "/app/uploads")
    OUTPUT_IMAGES_DIR = os.getenv("OUTPUT_IMAGES_DIR", "/app/output_images")

    os.makedirs(UPLOADS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)


# Global EasyOCR Reader with offline model path
MODEL_DIR = "./models"
GPU_ACCESS = False
if torch.cuda.is_available():
    print("CUDA is available")
    GPU_ACCESS = True
elif torch.backends.mps.is_available():
    print("MPS is available")
    GPU_ACCESS = True
else:
    print("only CPU is available")

reader = easyocr.Reader(
    ["en"], gpu=GPU_ACCESS, model_storage_directory=MODEL_DIR, download_enabled=False
)


client = MongoClient(Config.MONGO_URI, minPoolSize=10, maxPoolSize=20)
db = client["JioReimbursementDB"]
access_keys_collection = db["Access_token_details"]
collection = db["broadband"]
Ignore_Words = db["ignore_words_db"]

# config information
configCollection = db["config_info"]


# Define MongoDB Schema for Access Keys with `createdAt` and `updatedAt`
class AccessKeySchema(Schema):
    access_key = fields.Str(required=True)
    secret_key = fields.Str(required=True)
    channel_id = fields.Str(required=True)
    is_active = fields.Bool(required=True, default=True)
    createdAt = fields.DateTime(
        format="%Y-%m-%d %H:%M:%S", dump_only=True, missing=datetime.utcnow
    )
    updatedAt = fields.DateTime(
        format="%Y-%m-%d %H:%M:%S", dump_only=True, missing=datetime.utcnow
    )


access_key_schema = AccessKeySchema()


# Token authentication decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        print("Decorator initialized")
        token = request.headers.get("Authorization")
        channel_id_header = request.headers.get(
            "channel-id"
        )  # Extract channel_id from header

        logging.info(f"Token received: {token}")
        logging.info(f"Channel-ID from header: {channel_id_header}")

        if not token:
            logging.error("Token is missing!")
            return jsonify({"error": "Token is missing!"}), 403

        if not channel_id_header:
            logging.error("Channel-ID is missing in the header!")
            return jsonify({"error": "Channel-ID is missing in the header!"}), 403

        if token.startswith("Bearer "):
            token = token[7:]

        try:
            decrypted_token = decrypt_token(token)
            logging.info(f"Decrypted token successfully.")

            access_key = decrypted_token.get("access_key")
            channel_id_from_token = decrypted_token.get("channel_id")

            # Fetch the associated access key data from the database
            access_key_data = access_keys_collection.find_one(
                {
                    "access_key": access_key,
                    "is_active": True,  # Ensure the access key is active
                }
            )

            if not access_key_data:
                logging.error(f"Invalid or inactive access key: {access_key}")
                return jsonify({"error": "Invalid or inactive access key!"}), 403

            # Compare the channel_id from the header with the one in the DB
            channel_id_from_db = access_key_data.get("channel_id")
            if channel_id_header != channel_id_from_db:
                logging.error("Channel ID in header does not match with the database!")
                return (
                    jsonify(
                        {
                            "error": "Channel ID in header does not match with the database!"
                        }
                    ),
                    403,
                )

            # Compare the channel_id from token with the one in the DB
            if channel_id_from_token != channel_id_from_db:
                logging.error("Channel ID in token does not match with the database!")
                return (
                    jsonify(
                        {
                            "error": "Channel ID in token does not match with the database!"
                        }
                    ),
                    403,
                )

            # Check if token has expired
            now = datetime.utcnow()
            expiry_time = datetime.fromtimestamp(decrypted_token["exp"])
            if now > expiry_time:
                logging.error("Token has expired!")
                return jsonify({"error": "Token has expired!"}), 403

        except jwt.ExpiredSignatureError:
            logging.error("Token has expired!")
            return jsonify({"error": "Token has expired!"}), 403
        except jwt.InvalidTokenError:
            logging.error("Invalid token!")
            return jsonify({"error": "Invalid token!"}), 403
        except Exception as e:
            logging.error(f"Error in token decryption: {e}")
            return jsonify({"error": str(e)}), 403

        return f(*args, **kwargs)

    return decorated


def decrypt_token(enc_token):
    try:
        token_parts = enc_token.split(" ")
        if len(token_parts) > 1:
            enc_token = token_parts[1]
        reb64 = bytes.fromhex(enc_token)
        aes_key = base64.b64decode(Config.AES_KEY.encode())
        cipher = AES.new(aes_key, AES.MODE_EAX, nonce=reb64[:16])
        decrypted_token = cipher.decrypt_and_verify(reb64[16:-16], reb64[-16:])
        verified_token = jwt.decode(
            decrypted_token.decode(), Config.AES_KEY, algorithms=["HS256"]
        )
        return verified_token
    except Exception as e:
        logging.error(f"Error during token decryption: {e}")
        raise e


# Route to generate JWT token
@bills.route("/reimbursement/api/v1/token", methods=["POST"])
def generate_token():
    try:
        data = request.get_json()
        access_key = data.get("access_key")
        secret_key = data.get("secret_key")

        logging.info(f"Received access key: {access_key}")

        access_key_data = access_keys_collection.find_one(
            {"access_key": access_key, "secret_key": secret_key, "is_active": True}
        )

        if not access_key_data:
            logging.error("Invalid credentials provided.")
            response = OrderedDict(
                {
                    "status": 401,  # HTTP Status code for Unauthorized
                    "message": "Failed",
                    "data": {"message": "Token generation failed", "data": ""},
                }
            )
            return jsonify(response), 401

        channel_id = access_key_data.get("channel_id")
        expiry_time = datetime.utcnow() + timedelta(hours=8)

        payload = {
            "access_key": access_key,
            "channel_id": channel_id,
            "exp": expiry_time.timestamp(),
        }

        token = jwt.encode(payload, Config.AES_KEY, algorithm="HS256")
        aes_key = base64.urlsafe_b64decode(Config.AES_KEY.encode())[:32]
        cipher = AES.new(aes_key, AES.MODE_EAX)
        nonce = cipher.nonce
        ciphertext, tag = cipher.encrypt_and_digest(token.encode())

        encrypted_token = nonce + ciphertext + tag
        encrypted_token_hex = encrypted_token.hex()

        response_token = f"Bearer {encrypted_token_hex}"

        expiry_time = datetime.utcnow() + timedelta(hours=8)

        # Return the correctly ordered response
        response = OrderedDict(
            {
                "status": 200,  # HTTP Status code for Success
                "message": "Success",
                "data": {
                    "message": "Token for Broadband reimbursement generated successfully",
                    "expiry_time": expiry_time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "token": response_token,
                },
            }
        )

        logging.info(f"Generated token: {response_token}")
        return jsonify(response), 200
    except Exception as e:
        logging.error(f"Error generating token: {e}")
        response = OrderedDict(
            {
                "status": 500,  # HTTP Status code for Internal Server Error
                "message": "Failed",
                "data": {"message": str(e), "token": ""},
            }
        )
        return jsonify(response), 500


# Function to fetch words from MongoDB collection
def get_search_words_from_db():
    words = []
    for document in collection.find({}, {"_id": 0, "bills": 1}):
        words.append(document["bills"])
    return words


# Function to fetch maxPages and maxTime from MongoDB collection
def getMaxPagesAndMaxTimeAndOthersFromDB():
    global MAX_PAGES, MAX_TIMEOUT, MAX_DPI_RANGE, DPI_TO_SKIP, PYTESSERACT_CONFIG, CUSTOM_MESSAGES, NEED_TO_SENT_EMAIL, CUSTOM_REASONS, INITIAL_DPI
    for document in configCollection.find({}):
        if document["name"] == "MAX_PAGES":
            MAX_PAGES = int(document["value"])
        elif document["name"] == "MAX_TIMEOUT":
            MAX_TIMEOUT = float(document["value"])
        elif document["name"] == "MAX_DPI_RANGE":
            MAX_DPI_RANGE = int(document["value"])
        elif document["name"] == "DPI_TO_SKIP":
            DPI_TO_SKIP = int(document["value"])
        elif document["name"] == "INITIAL_DPI":
            INITIAL_DPI = int(document["value"])
        elif document["name"] == "PYTESSERACT_CONFIG":
            PYTESSERACT_CONFIG = str(document["value"])
        elif document["name"] == "CUSTOM_MESSAGES":
            CUSTOM_MESSAGES = list(document["value"])
        elif document["name"] == "NEED_TO_SENT_EMAIL":
            NEED_TO_SENT_EMAIL = list(document["value"])
        elif document["name"] == "CUSTOM_REASONS":
            CUSTOM_REASONS = list(document["value"])


# Function to fetch words from the ignore_words collection
def get_ignore_words_from_db():
    words = []
    for document in Ignore_Words.find({}, {"_id": 0, "ignore_words": 1}):
        words.append(document["ignore_words"])
    return words


# Function to check if the PDF is password-protected
def is_pdf_password_protected(pdf_file_bytes):
    try:
        # Try to open the PDF using PdfReader
        pdf_reader = PdfReader(io.BytesIO(pdf_file_bytes))
        # Check if the PDF is encrypted
        if pdf_reader.is_encrypted:
            try:
                # Try to decrypt with an empty password
                result = pdf_reader.decrypt("")
                if result == 0:
                    logging.info(
                        "PDF is password protected and cannot be opened with an empty password."
                    )
                    return True  # Still protected
                else:
                    logging.info(
                        "PDF was encrypted but successfully opened with empty password."
                    )
                    return False  # Opened without password, not 'protected' in the usual sense
            except Exception as e:
                logging.error(f"Decryption failed: {e}")
                return True
        return False  # Not encrypted
    except Exception as e:
        logging.error(f"Error checking if PDF is password protected: {e}")
        return False


# Function to extract images from PDF
def extract_images_from_pdf(pdf_file_bytes):
    if not pdf_file_bytes:
        raise Exception("The PDF file is empty or could not be read.")

    pages = []
    dpiPagesCache = {}

    for j in range(MAX_DPI_RANGE):
        dpi = j * DPI_TO_SKIP + INITIAL_DPI
        if dpi not in dpiPagesCache:
            # tempPages = convert_from_bytes(pdf_file_bytes, dpi)
            # if len(tempPages) >= MAX_PAGES:
            #     tempPages = tempPages[:MAX_PAGES]
            # dpiPagesCache[dpi] = tempPages
            try:
                tempPages = convert_from_bytes(pdf_file_bytes, dpi)
                if len(tempPages) >= MAX_PAGES:
                    tempPages = tempPages[:MAX_PAGES]
                dpiPagesCache[dpi] = tempPages
            except Exception as e:
                print(f"[ERROR] Failed to convert PDF at {dpi} DPI: {e}")
                continue  # skip this DPI and continue loop

        tempPages = dpiPagesCache[dpi]
        if len(tempPages) >= MAX_PAGES:
            pages += tempPages[:MAX_PAGES]
        else:
            pages += tempPages

    print("Pages: (New): ", pages)

    num_pages = len(pages) // MAX_DPI_RANGE
    images = [page for page in pages]
    logging.info(f"Extracted {num_pages} images from the PDF.")
    return images, num_pages


def preprocessingPilImageUsingOpenCV(pil_image):
    # Convert the PIL image to a NumPy array and change color space from RGB (PIL default) to BGR (OpenCV default)
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Convert the image to grayscale to simplify processing and reduce noise for OCR
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # specific kernel that enhances edges by amplifying the center pixel and subtracting the surrounding pixels.
    # Kernel: center has 9, the others are -1, effectively emphasizing contrast at edges.
    sharpenKernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    # Apply sharpening filter to the grayscale image
    sharpen = cv2.filter2D(gray, -1, sharpenKernel)

    # Apply binary inverse thresholding with Otsu's method:
    # - cv2.THRESH_BINARY_INV makes the text white on black background (helps Tesseract sometimes)
    # - cv2.THRESH_OTSU automatically determines the optimal threshold value
    # - `0` is a placeholder when using Otsu's method
    thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Return the processed binary image suitable for OCR
    return thresh


# Function to extract text from an image using OCR
# def extract_text_from_image(image):
#    # text = pytesseract.image_to_string(image, config="--oem 1 --psm 6")
#    text = pytesseract.image_to_string(
#        preprocessingPilImageUsingOpenCV(image), config=PYTESSERACT_CONFIG
#    )
#    new_text = "".join(re.findall(r"\w+", " ".join(text)))
#    # logging.info(f"OCR extracted text: {text[:100]}...")  # Log the first 100 characters of extracted text
#    # print("========================================================")
#    # print("Text: ", new_text.lower())
#    # print("========================================================")
#    return new_text.lower()

# Initialize EasyOCR Reader (you can set this up globally)
# reader = easyocr.Reader(['en'], gpu=True)


# Function to extract text from an image using EasyOCR
def extract_text_from_image(image):
    # If needed, preprocess the image (e.g., with OpenCV or PIL)
    processed_image = preprocessingPilImageUsingOpenCV(image)

    # Use EasyOCR to extract text
    results = reader.readtext(np.array(processed_image))

    # Combine all detected text into a single string
    text = " ".join([res[1] for res in results]).lower()

    # Normalize text: remove non-alphanumeric, lowercase
    # new_text = "".join(re.findall(r"\w+", text)).lower()

    return text
    # return new_text


# Function to search words in PDF
# Function to search words in PDF with better timeout handling
# async def search_words_in_pdf(
#     pdf_bytes, search_words, ignore_words, max_pages=MAX_PAGES, timeout=3600
# ):
#     try:
#         # Set a timeout for the entire function
#         return await asyncio.wait_for(
#             # _search_words_in_pdf_impl(pdf_bytes, search_words, ignore_words, max_pages),
#             timeout=timeout,
#         )
#     except asyncio.TimeoutError:
#         logging.error(f"PDF processing timed out after {timeout} seconds")
#         return False, False, True  # requires_physical_verification=True


# def sequentialImageTextExtractionFromPilObject(pdf_bytes, search_words, ignore_words):
#     text = _search_words_in_pdf_impl_2(pdf_bytes, search_words, ignore_words)
#     print(text)
#     return text


def call_inference_api(text, max_tokens=4096*4):
    payload = {
        # "TEXT": str("name Naveen date 08 sept 2025 bill jiofiber"),
        "TEXT": str(text),
        "MAX_OUTPUT_TOKENS": max_tokens,
    }

    try:
        response = requests.post(
            INFERENCE_API_URL, json=payload, timeout=120, verify=False
        )
        response.raise_for_status()  # raises HTTPError if not 2xx
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}
    except ValueError:  # JSON decoding failed
        return {"error": "Invalid JSON response", "raw": response.text}


def _search_words_in_pdf_impl_2(pdf_bytes, search_words, ignore_words, apiStartTime):

    images, num_pages = extract_images_from_pdf(pdf_bytes)
    isValidPdf = pdf_bytes.startswith(b"%PDF")

    if (isValidPdf == False) or num_pages == 0:
        logging.info("Unable to process the PDF file. It may be corrupted or invalid.")
        response = jsonify(
            {
                "message": str(CUSTOM_MESSAGES[0]),  # document not verified!!
                "status": 200,
                "isEmailNeeded": NEED_TO_SENT_EMAIL[0],
                "data": {
                    "reason": str(CUSTOM_REASONS[0]),  # unable to process the pdf file
                    "jio_broadband_bill": False,
                },
            }
        )
        return True, response, False, False, False

    diff = (datetime.now() - apiStartTime).total_seconds()
    if diff > MAX_TIMEOUT:
        logging.info(
            f"Document Not Verified (TIMEOUT: more than {MAX_TIMEOUT} seconds)"
        )
        response = jsonify(
            {
                "message": f"Document Not Verified (TIMEOUT: more than {MAX_TIMEOUT} seconds)",
                "status": 200,
                "isEmailNeeded": NEED_TO_SENT_EMAIL[1],
                "data": {
                    "reason": str(CUSTOM_REASONS[1]),  # physical verfication required
                    "jio_broadband_bill": False,
                },
            }
        )
        return True, response, False, False, False

    # totalPagesToScan = len(images)
    originalNumberOfPages = num_pages

    requires_physical_verification = originalNumberOfPages > MAX_PAGES
    images_to_scan = images

    word_found = False
    ignore_word_found = False

    isTimeoutReached = False
    timeoutResponse = 0

    id = 0
    for img in images_to_scan:
        diff = (datetime.now() - apiStartTime).total_seconds()
        if diff > MAX_TIMEOUT:
            isTimeoutReached = True
            timeoutResponse = jsonify(
                {
                    "message": f"Document Not Verified (TIMEOUT: more than {MAX_TIMEOUT} seconds)",
                    "status": 200,
                    "isEmailNeeded": NEED_TO_SENT_EMAIL[2],
                    "data": {
                        "reason": str(
                            CUSTOM_REASONS[2]
                        ),  # physical verfication required
                        "jio_broadband_bill": False,
                    },
                }
            )
            break

        text = extract_text_from_image(img)
        print(f"text: {text}")

        if id == 0:
            result = call_inference_api(text)
            print("===========================================================")
            print(result)
            print("===========================================================")

        id += 1
        print("-------------------------------------")
        print(f"[{id}] image: with text: {text}")
        print("-------------------------------------")
        # Check for ignored words first
        for word in ignore_words:
            if word in text:
                logging.info(f"Found ignored word: {word}. Document not verified.")
                ignore_word_found = True
                return False, 0, False, True, requires_physical_verification

        # If no ignored words, check for search words
        for word in search_words:
            if word in text:
                word_found = True
                logging.info(f"Found matching word: {word}. Document verified.")
                return False, 0, True, False, requires_physical_verification

        diff = (datetime.now() - apiStartTime).total_seconds()
        print(f"[TIME DIFF NOW (in seconds)]: {diff}")

        if diff > MAX_TIMEOUT:
            isTimeoutReached = True
            timeoutResponse = jsonify(
                {
                    "message": f"Document Not Verified (TIMEOUT: more than {MAX_TIMEOUT} seconds)",
                    "status": 200,
                    "isEmailNeeded": NEED_TO_SENT_EMAIL[3],
                    "data": {
                        "reason": str(
                            CUSTOM_REASONS[3]
                        ),  # physical verfication required
                        "jio_broadband_bill": False,
                    },
                }
            )
            break

    if isTimeoutReached == True:
        return True, timeoutResponse, False, False, False

    return False, 0, word_found, ignore_word_found, requires_physical_verification


# # Improved async OCR function with individual timeout
# async def extract_text_from_image_async(image, timeout=3600):
#     try:
#         # Use asyncio.to_thread with timeout for each image
#         return await asyncio.wait_for(
#             asyncio.to_thread(extract_text_from_image, image), timeout=timeout
#         )
#     except asyncio.TimeoutError:
#         logging.warning(f"OCR timed out for an image after {timeout} seconds")
#         return ""  # Return empty string on timeout to continue processing
#     except Exception as e:
#         logging.error(f"OCR exception: {e}")
#         return ""


# @creds: AI
def trim_pdf_base64(base64_pdf: str, keep_pages: int = MAX_PAGES) -> str:
    # Step 1: Decode base64 to PDF bytes
    pdf_bytes = base64.b64decode(base64_pdf)
    # Step 2: Read the PDF from bytes
    reader = PdfReader(BytesIO(pdf_bytes))
    writer = PdfWriter()
    # Step 3: Add only the first `keep_pages` pages
    for i in range(min(keep_pages, len(reader.pages))):
        writer.add_page(reader.pages[i])
    # Step 4: Write the trimmed PDF to a byte buffer
    output_stream = BytesIO()
    writer.write(output_stream)
    trimmed_pdf_bytes = output_stream.getvalue()
    # Step 5: Encode trimmed PDF back to base64
    trimmed_pdf_base64 = base64.b64encode(trimmed_pdf_bytes).decode("utf-8")
    return trimmed_pdf_base64


def decodeBase64WhileCheckingPadding(data) -> bytes:
    if isinstance(data, str):
        data = data.strip()
        missing_padding = len(data) % 4
        if missing_padding:
            data += "=" * (4 - missing_padding)
        return base64.b64decode(data, validate=True)

    return data


#################################################################
import pdfplumber

# from PIL import Image
# from datasets import Dataset
import torch
from transformers import (
    AutoProcessor,
    AutoModelForTokenClassification,
    # TrainingArguments,
    # Trainer,
    # LayoutLMv3Processor,
    # LayoutLMv3ForTokenClassification,
)

model_dir = "./layoutlmv3-finetune"
processor = AutoProcessor.from_pretrained(model_dir, apply_ocr=False)
model = AutoModelForTokenClassification.from_pretrained(model_dir)
# MODEL SET to EVALUATION
model = model.to("mps")
model.eval()


@bills.route("/reimbursement/validate", methods=["POST"])
@token_required
def bill_validate():
    try:
        pdf_file = request.files.get("pdf")

        if not pdf_file:
            logging.error("PDF file is required")
            response = {"message": "PDF file is required", "status": 400, "data": {}}
            return jsonify(response), 200

        pdf_bytes: bytes = pdf_file.stream.read()

        if not pdf_bytes:
            logging.error("PDF file is empty")
            response = {"message": "PDF file is empty", "status": 400, "data": {}}
            return jsonify(response), 200

        # Check if the PDF is password protected
        if is_pdf_password_protected(pdf_bytes):
            logging.info("PDF is password protected.")
            response = jsonify(
                {
                    "message": str(CUSTOM_MESSAGES[4]),  # document not verified
                    "status": 200,
                    "isEmailNeeded": NEED_TO_SENT_EMAIL[4],
                    "data": {
                        "reason": str(
                            CUSTOM_REASONS[4]
                        ),  # file is password-protected!!
                        "jio_broadband_bill": False,
                    },
                }
            )
            return response

    except Exception as e:
        logging.exception(f"Unexpected error while processing PDF: {str(e)}")
        response = {
            "message": "An error occurred while processing the PDF",
            "status": 500,
            "data": {"error": str(e)},
        }
        return jsonify(response), 200

    apiStartTime = datetime.now()
    print("apiStartTime: ", apiStartTime)

    # config info
    getMaxPagesAndMaxTimeAndOthersFromDB()

    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        writer = PdfWriter()
    except Exception as e:
        logging.error(f"Error during PDF processing: {e}")
        response = jsonify(
            {
                "status": 500,
                "message": str(e),
                "isEmailNeeded": NEED_TO_SENT_EMAIL[5],
                "data": {"reason": str(CUSTOM_REASONS[5]), "jio_broadband_bill": False},
            }
        )
        return response

    print("Total Pages Found : ", len(reader.pages))
    words = []
    bboxes = []

    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        max_pages = min(len(reader.pages), MAX_PAGES)
        for page_num, page in enumerate(pdf.pages[:max_pages]):
            img = page.to_image(resolution=300).original

            for word in page.extract_words():
                words.append(word["text"])
                x0 = int(word["x0"] / page.width * 1000)
                y0 = int(word["top"] / page.height * 1000)
                x1 = int(word["x1"] / page.width * 1000)
                y1 = int(word["bottom"] / page.height * 1000)
                bboxes.append([x0, y0, x1, y1])
            page = pdf.pages[0]
            img = page.to_image(resolution=300).original

    inputs = processor(
        images=img,
        text=words,
        boxes=bboxes,
        return_tensors="pt",
        # fp16='cpu',
        padding="max_length",
        truncation=True,
    )

    device = next(model.parameters()).device
    print(f"device: {device}")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()

    id = 1
    labels = []
    text = ""
    for word, label_id, bbox in zip(words, predictions, bboxes):
        print(f"id: [{id}] -> {word} | Label: {label_id} | BBox: {bbox}")
        labels.append(word)
        text += f"{word} "
        id += 1

    print(labels)
    print(text)

    # inferred_response = {}
    inferred_response = call_inference_api(text)

    return jsonify(
        {"status": 500, "data": labels, "inferred_response": inferred_response}
    )


#################################################################


# Route to search the PDF file
@bills.route("/reimbursement/broadband", methods=["POST"])
@token_required
def search_pdf():
    try:
        pdf_file = request.files.get("pdf")

        if not pdf_file:
            logging.error("PDF file is required")
            response = {"message": "PDF file is required", "status": 400, "data": {}}
            return jsonify(response), 200

        pdf_bytes: bytes = pdf_file.stream.read()

        if not pdf_bytes:
            logging.error("PDF file is empty")
            response = {"message": "PDF file is empty", "status": 400, "data": {}}
            return jsonify(response), 200

        # Check if the PDF is password protected
        if is_pdf_password_protected(pdf_bytes):
            logging.info("PDF is password protected.")
            response = jsonify(
                {
                    "message": str(CUSTOM_MESSAGES[4]),  # document not verified
                    "status": 200,
                    "isEmailNeeded": NEED_TO_SENT_EMAIL[4],
                    "data": {
                        "reason": str(
                            CUSTOM_REASONS[4]
                        ),  # file is password-protected!!
                        "jio_broadband_bill": False,
                    },
                }
            )
            return response

    except Exception as e:
        logging.exception(f"Unexpected error while processing PDF: {str(e)}")
        response = {
            "message": "An error occurred while processing the PDF",
            "status": 500,
            "data": {"error": str(e)},
        }
        return jsonify(response), 200

    apiStartTime = datetime.now()
    print("apiStartTime: ", apiStartTime)

    # config info
    getMaxPagesAndMaxTimeAndOthersFromDB()

    # pdf_file = request.files.get("pdf")
    #
    # if not pdf_file:
    #     logging.error("PDF file is required")
    #     response = {"message": "PDF file is required", "status": 400, "data": {}}
    #     return jsonify(response), 200
    #
    # pdf_bytes: bytes = pdf_file.stream.read()
    #
    # if not pdf_bytes:
    #     logging.error("PDF file is empty")
    #     response = {"message": "PDF file is empty", "status": 400, "data": {}}
    #     return jsonify(response), 200
    #
    # # Check if the PDF is password protected
    # if is_pdf_password_protected(pdf_bytes):
    #     logging.info("PDF is password protected.")
    #     response = jsonify(
    #         {
    #             "message": str(CUSTOM_MESSAGES[4]),  # document not verified
    #             "status": 200,
    #             "isEmailNeeded": NEED_TO_SENT_EMAIL[4],
    #             "data": {
    #                 "reason": str(CUSTOM_REASONS[4]),  # file is password-protected!!
    #                 "jio_broadband_bill": False,
    #             },
    #         }
    #     )
    #     return response

    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        writer = PdfWriter()
    except Exception as e:
        logging.error(f"Error during PDF processing: {e}")
        response = jsonify(
            {
                "status": 500,
                "message": str(e),
                "isEmailNeeded": NEED_TO_SENT_EMAIL[5],
                "data": {"reason": str(CUSTOM_REASONS[5]), "jio_broadband_bill": False},
            }
        )
        return response

    print("Total Pages Found : ", len(reader.pages))

    needPhysicalVerification = False
    if len(reader.pages) > MAX_PAGES:
        needPhysicalVerification = True

    for i in range(min(MAX_PAGES, len(reader.pages))):
        try:
            writer.add_page(reader.pages[i])
        except AssertionError:
            print(f"error in {i}")

    try:
        base64PDF: bytes = decodeBase64WhileCheckingPadding(pdf_bytes)
    except Exception as e:
        logging.error(f"Error during PDF processing: {e}")
        response = jsonify(
            {
                "status": 500,
                "message": str(e),
                "isEmailNeeded": NEED_TO_SENT_EMAIL[5],
                "data": {"reason": str(CUSTOM_REASONS[5]), "jio_broadband_bill": False},
            }
        )
        return response

    print("LAST: ", base64PDF[-20:])

    search_words = get_search_words_from_db()
    ignore_words = get_ignore_words_from_db()

    if len(search_words) == 0:
        logging.error("No search words found in broadband database")
        response = {
            "message": "Required keyword argument not found",
            "status": 404,
            "data": {},
        }
        return jsonify(response), 404

    if len(ignore_words) == 0:
        logging.error("No ignore words found in ignore_words database")
        response = {
            "message": "Ignore words not found in database",
            "status": 404,
            "data": {},
        }
        return jsonify(response), 404

    diff = (datetime.now() - apiStartTime).total_seconds()
    print(f"[TIME DIFF NOW (in seconds)]: {diff}")

    if diff > MAX_TIMEOUT:
        logging.info(
            f"Document Not Verified (TIMEOUT: more than {MAX_TIMEOUT} seconds)"
        )
        response = jsonify(
            {
                "message": f"Document Not Verified (TIMEOUT: more than {MAX_TIMEOUT} seconds)",
                "status": 200,
                "isEmailNeeded": NEED_TO_SENT_EMAIL[6],
                "data": {
                    "reason": str(CUSTOM_REASONS[6]),  # physical verfication required
                    "jio_broadband_bill": False,
                },
            }
        )
        return response

    (
        isTtimeOutReached,
        timeOutResponse,
        word_matched,
        ignore_word_found,
        requires_physical_verification,
    ) = _search_words_in_pdf_impl_2(pdf_bytes, search_words, ignore_words, apiStartTime)

    requires_physical_verification = needPhysicalVerification

    if isTtimeOutReached == True:
        logging.info(
            f"Document Not Verified (TIMEOUT: more than {MAX_TIMEOUT} seconds)"
        )
        response = timeOutResponse
        return response

    try:
        if ignore_word_found == True:
            logging.info("Found a word from the ignore list. Document not verified.")
            response = jsonify(
                {
                    "message": str(CUSTOM_MESSAGES[7]),
                    "status": 200,
                    "isEmailNeeded": NEED_TO_SENT_EMAIL[7],
                    "data": {
                        "reason": str(CUSTOM_REASONS[7]),
                        "jio_broadband_bill": False,
                    },
                }
            )
        elif word_matched == True:
            logging.info(
                "Found a matching word in the broadband table within the first 4 pages."
            )
            response = jsonify(
                {
                    "message": str(CUSTOM_MESSAGES[8]),
                    "status": 200,
                    "isEmailNeeded": NEED_TO_SENT_EMAIL[8],
                    "data": {
                        "reason": str(CUSTOM_REASONS[8]),  # Success
                        "jio_broadband_bill": True,
                    },
                }
            )
        elif requires_physical_verification:
            logging.info("PDF requires physical verification.")
            response = jsonify(
                {
                    "message": str(CUSTOM_MESSAGES[9]),  # document not verified
                    "status": 200,
                    "isEmailNeeded": NEED_TO_SENT_EMAIL[9],
                    "data": {
                        "reason": str(
                            CUSTOM_REASONS[9]
                        ),  # pdf requires physical verification
                        "jio_broadband_bill": False,
                    },
                }
            )
        else:
            logging.info("No matching words found. Document not verified.")
            response = jsonify(
                {
                    "message": str(CUSTOM_MESSAGES[10]),
                    "status": 200,
                    "isEmailNeeded": NEED_TO_SENT_EMAIL[10],
                    "data": {
                        "reason": str(
                            CUSTOM_REASONS[10]
                        ),  # no words found, need physical verfication
                        "jio_broadband_bill": False,
                    },
                }
            )

    except asyncio.TimeoutError:
        logging.error("Process timed out!")
        response = jsonify(
            {"message": "Process gateway timeout", "status": 504, "data": {}}
        )

    except Exception as e:
        logging.error(f"Error during PDF processing: {e}")
        response = jsonify(
            {
                "status": 500,
                "message": str(CUSTOM_MESSAGES[11]),
                "isEmailNeeded": NEED_TO_SENT_EMAIL[11],
                "data": {"reason": CUSTOM_REASONS[11], "jio_broadband_bill": False},
            }
        )

    return response


# Health check route
@bills.route("/reimbursement/health-check", methods=["GET"])
def health_check():
    logging.info("Health check endpoint hit.")
    return "Success"


if __name__ == "__main__":
    bills.run(host="0.0.0.0", port=5000, debug=True)



