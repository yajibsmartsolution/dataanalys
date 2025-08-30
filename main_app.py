import streamlit as st
import os

# Set page config
st.set_page_config(
    page_title="Grid Management System",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Simple authentication
def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if (st.session_state["username"] in st.secrets["passwords"] and
            st.session_state["password"] == st.secrets["passwords"][st.session_state["username"]]):
            st.session_state["password_correct"] = True
            st.session_state["user_role"] = st.secrets["passwords"][st.session_state["username"] + "_role"]
            del st.session_state["password"]  # Don't store the password
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 User not known or password incorrect")
        return False
    else:
        # Password correct.
        return True

# Main app
def main():
    st.title("🏭 Grid Management System Portal")
    
    if check_password():
        user_role = st.session_state.user_role
        
        st.success(f"Welcome! You are logged in as a {user_role}")
        
        if user_role == "operator":
            st.subheader("Injection Substation Application")
            st.write("Access the operator interface for real-time monitoring and control")
            if st.button("Launch Injection Substation", type="primary"):
                st.switch_page("apps/injection_substation.py")
                
        elif user_role == "analyst":
            st.subheader("YAJIB SMART Analytics Application")
            st.write("Access the analyst interface for data analysis and reporting")
            if st.button("Launch YAJIB SMART", type="primary"):
                st.switch_page("apps/yajib_smart.py")
                
        elif user_role == "admin":
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Injection Substation")
                st.write("Operator interface for real-time monitoring")
                if st.button("Launch Injection Substation", key="substation"):
                    st.switch_page("apps/injection_substation.py")
            with col2:
                st.subheader("YAJIB SMART Analytics")
                st.write("Analyst interface for data analysis")
                if st.button("Launch YAJIB SMART", key="yajib"):
                    st.switch_page("apps/yajib_smart.py")

if __name__ == "__main__":
    main()