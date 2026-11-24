import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd


# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="AgroVision AI | Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# Session State
# --------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "users" not in st.session_state:
    st.session_state.users = {}
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "current_user" not in st.session_state:
    st.session_state.current_user = None

def go_to(page):
    st.session_state.page = page
    st.rerun()

# --------------------------------------------------
# CSS STYLING
# --------------------------------------------------
st.markdown("""
<style>
/* Hide Streamlit Header (Deploy button, menu) */
header[data-testid="stHeader"] {
    display: none !important;
}

/* Adjust Top Padding so Navbar is fully visible */
.block-container {
    padding-top: 2.5rem !important;
}

/* Background */
.stApp {
    background: #0b110e; /* Dark theme background */
    color: #f8fafc;
}

/* Navbar */
.nav-title {
    font-size: 26px;
    font-weight: 800;
    color: #22c55e;
    margin-bottom: 0px !important;
}

/* Buttons */
button[kind="secondary"] {
    border: none !important;
    color: #4ade80 !important;
    background: rgba(34, 197, 94, 0.1) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease-in-out !important;
}
button[kind="secondary"]:hover {
    background: rgba(34, 197, 94, 0.2) !important;
    color: #22c55e !important;
    transform: translateY(-2px);
}

/* Primary Button */
button[kind="primary"] {
    background: linear-gradient(135deg, #22c55e, #16a34a) !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    color: #0b110e !important;
    border: none !important;
    transition: all 0.3s ease-in-out !important;
}
button[kind="primary"]:hover {
    box-shadow: 0 4px 15px rgba(34, 197, 94, 0.4) !important;
    transform: translateY(-2px);
}

/* Interactive Cards */
.feature-card {
    background: linear-gradient(135deg, #14532d, #064e3b);
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #22c55e;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    margin-bottom: 15px;
    height: 260px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}
.feature-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 10px 25px rgba(34, 197, 94, 0.3);
}
.feature-card h3 {
    color: #4ade80;
    margin-top: 15px;
    font-size: 22px;
}
.feature-card p {
    color: #d1fae5;
    font-size: 14px;
}

/* Disease Card (RED ALERT) */
.card-alert {
    background: linear-gradient(135deg, #7f1d1d, #450a0a);
    padding: 30px;
    border-radius: 18px;
    color: #fee2e2;
    text-align: center;
    font-size: 22px;
    font-weight: 700;
    box-shadow: 0 0 25px rgba(225, 29, 72, 0.4);
    border: 1px solid #ef4444;
    animation: fadeIn 0.6s ease-in-out;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Form Styling */
div[data-testid="stForm"] {
    background-color: rgba(6, 78, 59, 0.2);
    border: 1px solid #22c55e;
    border-radius: 15px;
    padding: 25px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}

/* Dashboard Summary Card */
.dashboard-summary {
    background: rgba(34, 197, 94, 0.05);
    border: 1px solid #22c55e;
    border-radius: 15px;
    padding: 20px;
}

/* Footer */
.footer {
    text-align: center;
    color: #94a3b8;
    margin-top: 50px;
    padding-top: 20px;
    border-top: 1px solid #1e293b;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# NAVBAR
# --------------------------------------------------
nav_cols = st.columns([3, 1, 1, 1, 1, 1])
with nav_cols[0]:
    st.markdown("<div class='nav-title'>🌿 AgroVision AI</div>", unsafe_allow_html=True)
with nav_cols[1]:
    if st.button("Home", use_container_width=True): go_to("Home")
with nav_cols[2]:
    if st.button("Features", use_container_width=True): go_to("Features")
with nav_cols[3]:
    if st.button("Contact", use_container_width=True): go_to("Contact")
with nav_cols[4]:
    if st.session_state.logged_in:
        if st.button("Dashboard", use_container_width=True): go_to("Dashboard")
    else:
        if st.button("Login", use_container_width=True): go_to("Login")
with nav_cols[5]:
    if st.session_state.logged_in:
        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.current_user = None
            go_to("Home")
    else:
        if st.button("Sign Up", use_container_width=True): go_to("SignUp")

st.markdown("<hr style='margin-top: 0px; border-color: #1e293b;'>", unsafe_allow_html=True)

# --------------------------------------------------
# Helpers for ML
# --------------------------------------------------
def format_label(label):
    return label.replace("___", " – ").replace("_", " ")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_plant_disease_model.keras")

model = load_model()

def model_prediction(img):
    image = tf.keras.preprocessing.image.load_img(img, target_size=(128, 128))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    preds = model.predict(image)
    return np.argmax(preds)

# --------------------------------------------------
# HOME / FEATURES / CONTACT PAGES
# --------------------------------------------------
if st.session_state.page in ["Home", "Features", "Contact"]:

    # HERO SECTION
    if st.session_state.page == "Home":
        if st.session_state.logged_in:
            # Layout for logged in users: Hero on left, Dashboard Summary on right
            col1, col2, col3 = st.columns([1.2, 0.1, 0.8])
            with col1:
                st.markdown(f"""
                <h1 style="color:#22c55e; font-size:48px; font-weight:800; line-height: 1.2;">
                    Welcome back, {st.session_state.current_user}!
                </h1>
                <p style="color:#cbd5e1; font-size:18px; margin-bottom: 30px;">
                    Analyze crop images instantly to detect diseases and get actionable recommendations.
                </p>
                """, unsafe_allow_html=True)
                if st.button("🌱 Detect Plant Disease", type="primary"):
                    go_to("Detect")
            with col3:
                st.markdown("<div class='dashboard-summary'>", unsafe_allow_html=True)
                st.markdown("<h3 style='color:#22c55e; margin-top:0;'>Dashboard Summary</h3>", unsafe_allow_html=True)
                user_data = st.session_state.users.get(st.session_state.current_user, {})
                if not isinstance(user_data, dict):
                    user_data = {"budget": 0, "profit_loss": 0}
                
                budget = user_data.get('budget', 0)
                pl = user_data.get('profit_loss', 0)
                
                st.metric("Total Budget", f"${budget:,.2f}")
                st.metric("Est. Profit/Loss", f"${pl:,.2f}", f"{(pl/budget)*100:.1f}%" if budget else "0%")
                
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Go to Full Dashboard ➔", use_container_width=True):
                    go_to("Dashboard")
                st.markdown("</div>", unsafe_allow_html=True)
                
            st.markdown("<br><br>", unsafe_allow_html=True)

        else:
            # Original Layout for logged out users
            col1, col2 = st.columns([1.2, 1])
            with col1:
                st.markdown("""
                <h1 style="color:#22c55e; font-size:54px; font-weight:800; line-height: 1.2;">
                    Early Plant Disease Detection
                </h1>
                <p style="color:#cbd5e1; font-size:18px; margin-bottom: 30px;">
                    AI-powered solution to detect crop diseases early, provide smart recommendations, 
                    and help farmers improve crop health and agricultural productivity.
                </p>
                """, unsafe_allow_html=True)
                if st.button("🌱 Detect Plant Disease", type="primary"):
                    st.warning("Please login to use this feature.")
                    import time
                    time.sleep(1)
                    go_to("Login")
            with col2:
                st.image("https://images.unsplash.com/photo-1625246333195-78d9c38ad449?q=80&w=1000&auto=format&fit=crop", use_container_width=True)

            st.markdown("<br><br>", unsafe_allow_html=True)

    # INTERACTIVE CARDS SECTION
    if st.session_state.page in ["Home", "Features"]:
        st.markdown("<h2 style='text-align: center; color: #22c55e;'>Explore Our Platform</h2>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        
        cards_data = [
            {"col": c1, "icon": "👨‍🌾", "title": "Hire Expert", "desc": "Connect with agriculture experts for guidance."},
            {"col": c2, "icon": "🌾", "title": "Sell Crop", "desc": "Platform for farmers to sell their crops easily."},
            {"col": c3, "icon": "🌱", "title": "Buy Seed", "desc": "Purchase high-quality seeds and farming products."},
            {"col": c4, "icon": "💬", "title": "Help", "desc": "Get support, FAQs, and assistance."}
        ]
        
        for c in cards_data:
            with c["col"]:
                st.markdown(f"""
                <div class="feature-card">
                    <div style="font-size: 50px;">{c['icon']}</div>
                    <h3>{c['title']}</h3>
                    <p>{c['desc']}</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"Go to {c['title']}", use_container_width=True, key=f"btn_{c['title']}"):
                    st.info(f"{c['title']} feature is coming soon!")

        st.markdown("<br><br>", unsafe_allow_html=True)

    # FEATURES HIGHLIGHT SECTION
    if st.session_state.page in ["Home", "Features"]:
        st.markdown("<h2 style='text-align: center; color: #22c55e;'>Why Choose AgroVision?</h2>", unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; color: #cbd5e1; font-size: 18px; max-width: 800px; margin: 0 auto;">
            Our state-of-the-art AI technology scans plant leaves to identify diseases instantly. 
            Receive actionable recommendations and expert guidance to protect your yield, maximize your farming potential, and secure your livelihood.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br><br>", unsafe_allow_html=True)

    # CONTACT SECTION
    if st.session_state.page in ["Home", "Contact"]:
        st.markdown("<h2 style='text-align: center; color: #22c55e;'>Contact Us</h2>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style='text-align:center; color:#94a3b8; margin-bottom:20px;'>
                Have questions? Send us a message and our team will get back to you! <br>
                📧 support@agroVisionAI.com &nbsp;&nbsp;|&nbsp;&nbsp; 📞 +91 723 702 4125
            </div>
            """, unsafe_allow_html=True)
            with st.form("contact_form"):
                st.text_input("Name")
                st.text_input("Email")
                st.text_area("Message")
                submitted = st.form_submit_button("Send Message", use_container_width=True)
                if submitted:
                    st.success("Message sent successfully! We will contact you soon.")

# --------------------------------------------------
# DETECT PAGE
# --------------------------------------------------
elif st.session_state.page == "Detect":
    st.markdown("<h1 style='color:#22c55e;'>🌿 Plant Disease Recognition</h1>", unsafe_allow_html=True)
    st.write("Upload a plant leaf image to detect disease")

    col1, col2 = st.columns(2)
    with col1:
        img = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if img:
            st.image(img, use_container_width=True)

    with col2:
        if img and st.button("🔍 Predict Disease", type="primary"):
            with st.spinner("Analyzing image..."):
                idx = model_prediction(img)

            class_names = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                'Cherry_(including_sour)___healthy',
                'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
                'Corn_(maize)___healthy', 'Grape___Black_rot',
                'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)',
                'Peach___Bacterial_spot', 'Peach___healthy',
                'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                'Raspberry___healthy', 'Soybean___healthy',
                'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
                'Strawberry___healthy', 'Tomato___Bacterial_spot',
                'Tomato___Early_blight', 'Tomato___Late_blight',
                'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
                'Tomato___Spider_mites Two-spotted_spider_mite',
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
            ]

            disease = format_label(class_names[idx])

            st.markdown(f"""
            <div class="card-alert">
                🚨 Disease Detected <br><br>
                {disease}
            </div>
            """, unsafe_allow_html=True)

# --------------------------------------------------
# DASHBOARD PAGE
# --------------------------------------------------
elif st.session_state.page == "Dashboard" and st.session_state.logged_in:
    username = st.session_state.current_user
    user_data = st.session_state.users.get(username, {})
    
    # Fallback if old account
    if not isinstance(user_data, dict):
        user_data = {"photo": None, "gender": "N/A", "land_area": 0, "crops": [], "budget": 0, "profit_loss": 0}

    st.markdown(f"<h1 style='color:#22c55e;'>👨‍🌾 Dashboard - {username}</h1>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color: #1e293b;'>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Profile Information")
        if user_data.get("photo"):
            st.image(user_data["photo"], width=200)
        else:
            st.markdown("🧑‍🌾 *No photo uploaded*")
        
        st.write(f"**Gender:** {user_data.get('gender', 'N/A')}")
        st.write(f"**Land Area:** {user_data.get('land_area', 0)} Acres")
        
        st.markdown("<br>### Crops Farmed", unsafe_allow_html=True)
        crops = user_data.get("crops", [])
        if crops:
            for c in crops:
                st.markdown(f"- 🌾 {c}")
        else:
            st.write("*No crops listed.*")
            
    with col2:
        st.markdown("### Investment & Financials")
        
        budget = user_data.get("budget", 0.0)
        profit_loss = user_data.get("profit_loss", 0.0)
        
        # Using Streamlit metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Budget", f"${budget:,.2f}")
        m2.metric("Est. Profit/Loss", f"${profit_loss:,.2f}", f"{(profit_loss/budget)*100:.1f}%" if budget else "0%")
        m3.metric("Current Balance", f"${(budget + profit_loss):,.2f}")
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("#### Crop Insights & Market Trend")
        st.info("Based on your listed crops, current market rates are stable. Profit estimation is projected at +15% for the current harvest cycle.")
        
        if budget > 0:
            st.markdown("#### Portfolio Growth Projection")
            # Simulated fake data chart
            chart_data = pd.DataFrame(
                np.random.randn(20, 1) * (budget * 0.02) + (profit_loss / 20),
                columns=['Projected Value']
            ).cumsum() + budget
            st.line_chart(chart_data)

# --------------------------------------------------
# SIGN UP PAGE
# --------------------------------------------------
elif st.session_state.page == "SignUp":
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h2 style='text-align: center; color: #22c55e;'>Create an Account</h2>", unsafe_allow_html=True)
        with st.form("signup_form"):
            st.markdown("#### Login Credentials")
            new_user = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            
            st.markdown("#### Farming Profile")
            photo = st.file_uploader("Profile Photo", type=["jpg", "jpeg", "png"])
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            land_area = st.number_input("Land Area (Acres)", min_value=0.0, step=0.5)
            crops = st.text_input("Crops Farmed (comma separated, e.g., Wheat, Corn, Soybeans)")
            budget = st.number_input("Farming Budget ($)", min_value=0.0, step=100.0)

            submit_signup = st.form_submit_button("Sign Up", use_container_width=True)
            
            if submit_signup:
                if new_user in st.session_state.users:
                    st.error("Username already exists!")
                elif new_password != confirm_password:
                    st.error("Passwords do not match!")
                elif not new_user or not new_password:
                    st.error("Please fill all required fields (Username & Password).")
                else:
                    # Clean up crops list
                    crop_list = [c.strip() for c in crops.split(",") if c.strip()]
                    # Simulated profit of 15%
                    sim_profit = budget * 0.15

                    st.session_state.users[new_user] = {
                        "password": new_password,
                        "photo": photo,
                        "gender": gender,
                        "land_area": land_area,
                        "crops": crop_list,
                        "budget": budget,
                        "profit_loss": sim_profit
                    }
                    st.success("Account created successfully! Redirecting to login...")
                    import time
                    time.sleep(1.5)
                    go_to("Login")

# --------------------------------------------------
# LOGIN PAGE
# --------------------------------------------------
elif st.session_state.page == "Login":
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h2 style='text-align: center; color: #22c55e;'>Login</h2>", unsafe_allow_html=True)
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_login = st.form_submit_button("Login", use_container_width=True)
            
            if submit_login:
                user_record = st.session_state.users.get(username)
                
                is_valid = False
                if user_record:
                    # Handle both old string passwords and new dict profiles
                    if isinstance(user_record, dict):
                        is_valid = (user_record["password"] == password)
                    else:
                        is_valid = (user_record == password)

                if is_valid:
                    st.session_state.logged_in = True
                    st.session_state.current_user = username
                    st.success("Logged in successfully!")
                    import time
                    time.sleep(1)
                    go_to("Dashboard")
                else:
                    st.error("Invalid username or password.")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("""
<div class="footer">
    AgroVision – AI Powered Plant Disease Detection 🌱 <br>
    Developed by Sukhveer Singh & Jafae 
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    pass
