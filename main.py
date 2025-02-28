import os
import time
import random
import pickle
import json
import pandas as pd
import requests
import re
import isbnlib
import asyncio
from dotenv import load_dotenv
from flask import Flask, request, send_file, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from playwright.async_api import async_playwright
import isbnlib
from playwright.sync_api import sync_playwright  # Sync Playwright API
from playwright_stealth import stealth_sync  # Prevent Bot Detection
import easyocr
import cv2
from urllib.parse import urlencode
from werkzeug.utils import secure_filename
from cryptography.fernet import Fernet
import base64
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
from flask_cors import CORS  # Import CORS


# Load environment variables
load_dotenv()

# ScraperAPI Key
API_KEY = os.getenv("SCRAPPER_API_KEY")

# # Amazon Credentials
# AMAZON_USERNAME = os.getenv("AMAZON_USERNAME")
# AMAZON_PASSWORD = os.getenv("AMAZON_PASSWORD")

# Global dictionary to store Amazon credentials
# amazon_credentials = {}

# File paths
INPUT_EXCEL = "booksinput.xlsx"
OUTPUT_EXCEL = "booksoutput.xlsx"
UPLOAD_FOLDER = 'uploads'

# Initialize Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Enable CORS for all routes
CORS(app)

# Define the scheduler
scheduler = BackgroundScheduler()

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize EasyOCR
reader = easyocr.Reader(["en"])


# Encryption Key (Generate once and keep it safe)
KEY_FILE = "secret.key"
CREDENTIALS_FILE = "amazon_credentials.json"

# Function to generate an encryption key (only run once)
def generate_key():
    if not os.path.exists(KEY_FILE):
        key = Fernet.generate_key()
        with open(KEY_FILE, "wb") as key_file:
            key_file.write(key)
generate_key()

# Function to load the encryption key
def load_key():
    with open(KEY_FILE, "rb") as key_file:
        return key_file.read()

fernet = Fernet(load_key())

# Function to save encrypted credentials
def save_credentials(username, password):
    encrypted_data = {
        "username": base64.b64encode(fernet.encrypt(username.encode())).decode(),
        "password": base64.b64encode(fernet.encrypt(password.encode())).decode(),
    }
    with open(CREDENTIALS_FILE, "w") as f:
        json.dump(encrypted_data, f)

# Function to load credentials
def load_credentials():
    if os.path.exists(CREDENTIALS_FILE):
        with open(CREDENTIALS_FILE, "r") as f:
            encrypted_data = json.load(f)
            return {
                "username": fernet.decrypt(base64.b64decode(encrypted_data["username"])).decode(),
                "password": fernet.decrypt(base64.b64decode(encrypted_data["password"])).decode(),
            }
    return None


def random_sleep(min_delay=1, max_delay=3):
    """Randomized sleep to simulate human-like pauses."""
    delay = random.uniform(min_delay, max_delay)
    print(f"Waiting for {delay:.2f} seconds...")
    time.sleep(delay)

def clean_isbn(isbn):
    """Removes commas and extra spaces from ISBN."""
    return str(isbn).replace(",", "").strip()

def extract_author(brand):
    """Extracts author name from the 'brand' field."""
    if brand:
        return re.sub(r"by |\(Author\)", "", brand).strip()
    return "N/A"

def extract_edition(publisher):
    """Extracts edition information from the publisher field."""
    if not publisher or publisher == "N/A":
        return "N/A"
    
    match = re.search(r'(\b\w+ edition\b)', publisher, re.IGNORECASE)
    return match.group(1) if match else "N/A"

def extract_sub_publisher(publisher):
    """Extracts sub-publisher from the publisher field."""
    if not publisher or publisher == "N/A":
        return "N/A"
    
    parts = publisher.split(";")
    return parts[1].strip() if len(parts) > 1 else "N/A"

def get_book_data(isbn):
    """Fetches book data from Amazon using ScraperAPI."""
    payload = {
        'api_key': API_KEY,
        'asin': isbn,
        'country': 'us',
        'tld': 'com'
    }
    url = 'https://api.scraperapi.com/structured/amazon/product?' + urlencode(payload)

    response = requests.get(url)
    print(f"Fetching data for ISBN: {isbn} | Status Code: {response.status_code}")

    try:
        json_data = response.json() if response.status_code == 200 else None
        print(f"📜 JSON Response for ISBN {isbn}:")
        print(json.dumps(json_data, indent=4))
        return json_data
    except requests.exceptions.JSONDecodeError as e:
        print(f"JSONDecodeError for ISBN {isbn}: {e}")
        return None

def process_books(input_file, output_file):
    """Process books data from input Excel file."""
    try:
        df = pd.read_excel(input_file, dtype={'ISBN13': str})
        df['ISBN13'] = df['ISBN13'].astype(str).apply(clean_isbn)

        result_data = []

        for isbn in df['ISBN13']:
            isbn = clean_isbn(isbn)

            try:
                isbn_10 = isbnlib.to_isbn10(isbn)
                isbn_to_use = isbn_10
            except isbnlib.NotValidISBNError:
                isbn_to_use = isbn

            book_data = get_book_data(isbn_to_use)

            if book_data:
                product_info = book_data.get("product_information", {})

                result_data.append({
                    'ISBN13': product_info.get('isbn_13', isbn),
                    'Author': extract_author(book_data.get('brand', 'N/A')),
                    'Title': book_data.get('name', 'N/A'),
                    'Edition': extract_edition(product_info.get('publisher', 'N/A')),
                    'Pub': product_info.get('publisher', 'N/A'),
                    'Sub Pub': extract_sub_publisher(product_info.get('publisher', 'N/A')),
                    'Current_Price': book_data.get('pricing', 'N/A'),
                    'Amazon URL': f"https://www.amazon.com/dp/{isbn_to_use}"
                })
            else:
                print(f"Failed to fetch data for ISBN: {isbn}")

        output_df = pd.DataFrame(result_data)
        output_df.to_excel(output_file, index=False)
        print(f"Data updated and saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")
        raise

def setup_driver():
    """Sets up Playwright WebDriver with Sync API."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # You can change to `headless=True` for no GUI
        context = browser.new_context()  # Create a new browser context
        page = context.new_page()  # Open a new page within the context

        # Apply stealth mode to prevent bot detection
        stealth_sync (page)
        # return page, context, browser  # Return the browser as well for later management
        # page = browser.new_page()

        # Clear cookies at the start
        page.context.clear_cookies()
        print("All cookies deleted on startup.")
        return page


def detect_and_solve_captcha(page, max_attempts=5):
    """Detects and solves CAPTCHA if present, with retry mechanism."""
    attempts = 0

    while attempts < max_attempts:
        content = page.content().lower()
        if "captcha" in content:
            print(f"CAPTCHA detected (Attempt {attempts + 1}/{max_attempts}). Solving...")
            captcha_image = page.locator("img[src*='captcha']")
            
            if captcha_image:
                captcha_image.screenshot(path="captcha.png")
                solved_text = solve_captcha("captcha.png")

                if solved_text:
                    page.fill("input#captchacharacters", solved_text)
                    random_sleep(3, 5)
                    page.click("button[type='submit']")
                    random_sleep(3, 5)

                    # Check if CAPTCHA still exists (i.e., wrong input)
                    content_after_submission = page.content().lower()
                    if "captcha" not in content_after_submission:
                        print("✅ CAPTCHA solved successfully!")
                        return True
                    else:
                        print("CAPTCHA submission failed. Retrying...")
                        page.reload()
                        random_sleep(3, 5)
                        attempts += 1
                else:
                    print("CAPTCHA solving failed. Retrying...")
                    page.reload()
                    random_sleep(3, 5)
                    attempts += 1
            else:
                print("CAPTCHA image not found.")
                return False
        else:
            return True  # No CAPTCHA found, proceed

    print("Failed to solve CAPTCHA after multiple attempts. Manual intervention required.")
    return False


def solve_captcha(image_path):
    """Extracts CAPTCHA text using EasyOCR."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
        processed_path = "processed_captcha.png"
        cv2.imwrite(processed_path, img)

        result = reader.readtext(img, detail=0)
        captcha_text = " ".join(result).strip()
        print(f"CAPTCHA Solved: {captcha_text}")
        return captcha_text
    except Exception as e:
        print(f"CAPTCHA solving failed: {e}")
        return None

def add_to_cart(amazon_url, page, quantity, isbn, df_input):
    """Adds the product to the Amazon cart while maintaining session."""
    try:
        # 🔹 Load session cookies before going to the product page
        if os.path.exists("amazon_cookies.pkl"):
            cookies = pickle.load(open("amazon_cookies.pkl", "rb"))
            page.context.add_cookies(cookies)
            print("Loaded saved cookies.")

        page.goto(amazon_url)
        random_sleep(3, 4)

        if not detect_and_solve_captcha(page):
            return False

        # Reload the page to apply cookies
        page.reload()
        random_sleep(2, 3)

        # Select Quantity Dropdown
        # quantity_selector = page.query_selector("select#quantity")
        # if quantity_selector:
        #     page.select_option("select#quantity", str(quantity))  # Set the correct quantity
        #     print(f"Selected quantity: {quantity}")
        #     random_sleep(2, 3)

        try:
            page.wait_for_selector("select#quantity", timeout=10000)  # Wait for dropdown
            available_quantities = page.locator("select#quantity").evaluate("el => [...el.options].map(o => o.value)")

            if str(quantity) in available_quantities:
                page.select_option("select#quantity", str(quantity))
                quantity_added = quantity
                print("For 2th oneeeee",quantity_added)
            else:
                max_quantity = max(map(int, available_quantities))  # Select max available
                print(f"Requested {quantity}, but only {max_quantity} available. Selecting {max_quantity}.")
                page.select_option("select#quantity", str(max_quantity))
                quantity_added = max_quantity  # Using max available quantity
                print("For 6th oneeeee",quantity_added)

        except Exception as e:
            print(f"Could not select quantity: {e}")

    
        # Simulate cursor movement to the "Add to Cart" button
        add_to_cart_selectors = [
            "#add-to-cart-button",
            "input[name='submit.add-to-cart']",
            ".a-button-input[name='submit.add-to-cart']"
        ]

        for selector in add_to_cart_selectors:
            button = page.query_selector(selector)
            box = button.bounding_box()
            if box:  # Ensure bounding box exists
                center_x = box["x"] + box["width"] / 2
                center_y = box["y"] + box["height"] / 2
                page.mouse.move(center_x, center_y)
                random_sleep(1, 3)
                button.click()
                # Debug: Confirm value before writing to Excel
                print(f"Before updating Excel: ISBN = {isbn}, Quantity Added = {quantity_added}")
                # Update the input file (Excel) to mark it as added
                index = df_input[df_input['ISBN13'] == isbn].index
                df_input.at[index[0], "Added to Cart"] = "Yes"
                df_input.at[index[0], "Quantity Added"] = quantity_added  
                df_input.to_excel(INPUT_EXCEL, index=False)  # Save changes
                return True
            else:
                print("Could not find Add to Cart button coordinates.")
                return False

        print("Add to Cart button not found.")
        return False
    
    except Exception as e:
        print(f"Could not add product to cart: {e}")
        return False

def get_login_url(page):
    """Finds the Amazon login URL dynamically."""
    page.goto("https://www.amazon.com")
    random_sleep(3, 5)
    if not detect_and_solve_captcha(page):
        return False
    login_element = page.query_selector("a[href*='signin']")
    if login_element:
        login_url = login_element.get_attribute("href")
        print(f"Found login URL: {login_url}")
        return login_url
    else:
        print("Could not find login URL. Using fallback URL.")
        return "https://www.amazon.com/ap/signin"


# def amazon_login(page, username, password):
def amazon_login(page):
    """Logs in to Amazon and saves session cookies to avoid re-login."""
    print("Logging in to Amazon...")

    # 🔹 Delete old cookies (optional)
    if os.path.exists("amazon_cookies.pkl"):
        os.remove("amazon_cookies.pkl")
        print("Deleted old cookies.")

    # page.goto("https://www.amazon.in/ap/signin?openid.pape.max_auth_age=0&openid.return_to=https%3A%2F%2Fwww.amazon.in%2F%3Fref_%3Dnav_signin&openid.identity=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.assoc_handle=inflex&openid.mode=checkid_setup&openid.claimed_id=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.ns=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0")
    """Logs in to Amazon using dynamically found login URL."""
    login_url = get_login_url(page)
    page.goto(login_url, wait_until="load")
    random_sleep(5, 8)

    """Logs in to Amazon using stored credentials."""

    # Load credentials
    credentials = load_credentials()
    if not credentials:
        print("Amazon credentials are not found!")
        return False
    
    username = credentials["username"]
    password = credentials["password"]

    if not username or not password:
        print("Amazon credentials are not set!")
        return False
    
    # Wait for email field to be available
    try:
        page.wait_for_selector("#ap_email", timeout=60000)  # Wait for email input for up to 60 seconds
    except TimeoutError:
        print("Timeout: Email field (#ap_email) not found! Retrying...")
        page.reload(wait_until="load")  # Reload and retry
        random_sleep(5, 8)
        # return False
        try:
            page.wait_for_selector("#ap_email", timeout=30000)
        except TimeoutError:
            print("Email field not found after retry. Exiting login.")
            return False
    random_sleep(5, 8)


    if not detect_and_solve_captcha(page):
        return False
    

    # page.fill("input#ap_email", username)
    # Simulate human-like typing
    for char in username:
        page.fill("#ap_email", page.input_value("#ap_email") + char)
        random_sleep(0.2, 0.4)
    random_sleep(3, 4)
    page.click("input#continue")
    random_sleep(3, 4)


    if not detect_and_solve_captcha(page):
        return False


    # Wait for the password field
    try:
        page.wait_for_selector("#ap_password", timeout=60000)  # Wait for password field to appear
    except TimeoutError:
        print("Timeout: Password field (#ap_password) not found!")
        return False



    # page.fill("input#ap_password", password)
    for char in password:
        page.fill("#ap_password", page.input_value("#ap_password") + char)
        random_sleep(0.2, 0.4)
    random_sleep(3, 4)
    page.click("input#signInSubmit")
    random_sleep(5, 7)

    if not detect_and_solve_captcha(page):
        return False

    print("Logged in to Amazon successfully!")

    # 🔹 Save cookies after successful login
    cookies = page.context.cookies()
    pickle.dump(cookies, open("amazon_cookies.pkl", "wb"))
    print("New cookies saved successfully!")

def monitor_price_drops():
    """Monitors price drops and adds books to the cart if they meet criteria."""
    # df = pd.read_excel(OUTPUT_EXCEL)
        # Load input and output Excel files
    df_input = pd.read_excel(INPUT_EXCEL)   # Contains "Max Price"
    df_output = pd.read_excel(OUTPUT_EXCEL) # Contains "Current Price"


    # Normalize ISBN format (remove hyphens, extra spaces, and hidden Unicode characters)
    df_input['ISBN13'] = df_input['ISBN13'].astype(str).str.replace("-", "").str.strip()
    df_output['ISBN13'] = df_output['ISBN13'].astype(str).str.replace("-", "").str.strip()
    df_output['ISBN13'] = df_output['ISBN13'].apply(lambda x: x.encode('ascii', 'ignore').decode())  # Remove hidden chars

    # Normalize "Max Price" (remove currency symbols, convert to float)
    df_input['Max Price'] = df_input['Max Price'].astype(str).str.replace('₹', '').str.replace(',', '').str.strip()
    df_input['Max Price'] = pd.to_numeric(df_input['Max Price'], errors='coerce')

    # Merge both dataframes on ISBN13 (or another unique column)
    df = pd.merge(df_output, df_input[['ISBN13', 'Max Price',  'Quantity']], on='ISBN13', how='left')
    df = df.drop_duplicates(subset=['ISBN13'])
    # Rename columns to avoid _x and _y issues
    df = df.rename(columns={'Max Price_y': 'Max Price', 'Quantity_y': 'Quantity'})

    # Drop unnecessary duplicate columns if they exist
    df = df.drop(columns=[col for col in df.columns if col.endswith('_x')], errors='ignore')

    # Ensure "Added to Cart" column exists
    # Ensure "Added to Cart" and "Quantity Added" columns exist in df_input
    if "Added to Cart" not in df_input.columns:
        df_input["Added to Cart"] = "No"
    if "Quantity Added" not in df_input.columns:
        df_input["Quantity Added"] = 0

    # print(df.columns)
    print("Columns in final DataFrame:", df.columns.tolist())
    print("\nData in final DataFrame:\n", df.to_string(index=False))  

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()  # Create a persistent browser context
        page = context.new_page()  # Use the same session
        stealth_sync (page)  # Prevent Bot Detection

        # Step 1: Log in once
        # amazon_login(page, AMAZON_USERNAME, AMAZON_PASSWORD)
        amazon_login(page)

        for index, row in df.iterrows():
            amazon_url = row['Amazon URL']
            isbn = row['ISBN13']
            requested_quantity = int(row['Quantity'])

            # Get index in df_input for updating it later
            input_index = df_input[df_input["ISBN13"] == isbn].index


            # Check if this product has already been added to cart
            already_added = df_input.at[input_index[0], "Added to Cart"] == "Yes"
            quantity_added = df_input.at[input_index[0], "Quantity Added"] 

            # Ensure quantity_added is numeric for comparison
            if pd.isna(quantity_added):
                quantity_added = 0
            else:
                quantity_added = int(quantity_added)


            # Skip if the product has already been added with the requested quantity
            if already_added and quantity_added == requested_quantity:
                print(f"Skipping {row['Title']} (ISBN: {isbn}) - Already added to cart with requested quantity ({requested_quantity}).")
                continue

            # If some quantity was added but not the full requested amount, we still want to check price
            if already_added and quantity_added < requested_quantity:
                print(f"Partially added {row['Title']} (ISBN: {isbn}) - {quantity_added}/{requested_quantity} units. Checking if we can add more.")
                # We could try to add more if price is still good


            try:
                # last_price = float(str(row['Price']).replace('$', '').replace(',', '').strip())
                max_price = float(str(row['Max Price']).replace('$', '').replace(',', '').strip())
                current_price = float(str(row['Current_Price']).replace('$', '').replace(',', '').strip())
            except ValueError:
                print(f"Invalid price data for {row['Title']} (ISBN: {row['ISBN13']}). Skipping...")
                continue

            # if current_price and last_price:
            if max_price and current_price:
                if max_price >= current_price:
                    print(f"Price drop detected for {row['Title']}. Adding to cart...")
                    # add_to_cart(amazon_url, page, requested_quantity, isbn, df_input)
                    # If partially added, try to add the remaining quantity
                    remaining_quantity = requested_quantity - quantity_added if already_added else requested_quantity
                    
                    if remaining_quantity > 0:
                        if add_to_cart(amazon_url, page, remaining_quantity, isbn, df_input):
                            print(f"Successfully added {row['Title']} to cart.")
                        else:
                            print(f"Failed to add {row['Title']} to cart.")
                else:
                    print(f"ℹ️ No significant price drop for {row['Title']}.")
            # else:
            #     print(f"Invalid price data for {row['Title']}.")
            # Random wait time before checking next product
            random_sleep(10, 20)

        browser.close()  # Close browser after processing all products
    # Save updated output file
    df.to_excel(OUTPUT_EXCEL, index=False)
    print("Output file updated successfully!")


# Function to run process_books
def schedule_process_books():
    """This will be called by the scheduler to run every hour."""
    try:
        process_books(INPUT_EXCEL, OUTPUT_EXCEL)
        print("Process books completed successfully!")
    except Exception as e:
        print(f"Error while processing books: {str(e)}")

# Function to run monitor_price_drops
def schedule_monitor_price_drops():
    """This will be called by the scheduler to run every hour."""
    try:
        monitor_price_drops()
        print("Price drop monitoring completed successfully!")
    except Exception as e:
        print(f"Error while monitoring price drops: {str(e)}")

# Function to start the scheduler
def start_scheduler():
    # Schedule the tasks every 1 hour
    scheduler.add_job(schedule_process_books, 'interval', hours=1)
    scheduler.add_job(schedule_monitor_price_drops, 'interval', hours=1)

    # Start the scheduler
    scheduler.start()

# ----------------- Flask Routes -----------------

@app.route('/')
def home():
    return 'Flask App: Amazon Scrapper'


@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload an Excel file containing ISBNs."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Copy to input excel location
        import shutil
        shutil.copy(filepath, INPUT_EXCEL)
        
        return jsonify({'message': 'File uploaded successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/scrape', methods=['GET'])
def scrape_books():
    """Scrape book data based on ISBNs."""
    try:
        process_books(INPUT_EXCEL, OUTPUT_EXCEL)
        return jsonify({'message': 'Scraping completed successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/inputfile', methods=['GET'])
def get_input_file():
    """Retrieve the input Excel file."""
    try:
        if not os.path.exists(INPUT_EXCEL):
            return jsonify({'error': 'Input file not found'}), 404
        
        # Send the input file back to the client
        directory = os.path.dirname(INPUT_EXCEL)
        return send_from_directory(directory, os.path.basename(INPUT_EXCEL))
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/outputfile', methods=['GET'])
def get_output_file():
    """Return the output Excel file."""
    try:
        if os.path.exists(OUTPUT_EXCEL):
            return send_file(
                OUTPUT_EXCEL,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                as_attachment=True,
                download_name='booksoutput.xlsx'
            )
        else:
            return jsonify({'error': 'Output file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/set_amazon_credentials', methods=['POST'])
def set_amazon_credentials():
    """API to receive Amazon username and password from frontend and store them"""
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')

        if not username or not password:
            return jsonify({'error': 'Username and password are required'}), 400

        save_credentials(username, password)

        return jsonify({'message': 'Amazon credentials stored successfully'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_amazon_credentials', methods=['GET'])
def get_amazon_credentials():
    """API to retrieve stored Amazon credentials"""
    try:
        credentials = load_credentials()
        if credentials:
            return jsonify(credentials)
        else:  # No credentials stored
            return jsonify({'message': 'No Amazon credentials stored', 'status': 'error'}), 200
    except Exception as e:
        return jsonify({'message': 'Error occurred while retrieving credentials', 'error': str(e), 'status': 'error'}), 500


@app.route('/clear_amazon_credentials', methods=['GET'])
def clear_amazon_credentials():
    """API to clear stored Amazon credentials"""
    try:
        if os.path.exists(CREDENTIALS_FILE):
            os.remove(CREDENTIALS_FILE)  # Delete the credentials file
        return jsonify({'message': 'Amazon credentials cleared successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/automation', methods=['GET'])
def run_automation():
    """Run the price monitoring automation."""
    try:        
        # Run the function 
        monitor_price_drops()
        return jsonify({'message': 'Automation completed successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # start_scheduler()
    # app.run(debug=True)
    port = int(os.environ.get("PORT", 10000))  # Get the port from environment variable
    app.run(host='0.0.0.0', port=port, debug=True)