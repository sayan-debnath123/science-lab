import streamlit as st
import google.generativeai as genai
from PIL import Image
import time
import os
import tempfile

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Real-Time Science Lab Analyzer",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING ---
st.markdown("""
<style>
    .report-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #4CAF50;
    }
    .safety-warning {
        background-color: #ffcccc;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ff0000;
        color: #990000;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "analysis_context" not in st.session_state:
    st.session_state.analysis_context = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None

# --- SIDEBAR & SETTINGS ---
with st.sidebar:
    st.title("⚗️ Lab Assistant")
    
    # API Key Input
    api_key = st.text_input("Enter Google Gemini API Key", type="password")
    if api_key:
        genai.configure(api_key=api_key)
    
    st.divider()
    
    # Complexity Level
    complexity = st.select_slider(
        "Explanation Level",
        options=["Beginner", "Intermediate", "Advanced", "Researcher"],
        value="Intermediate"
    )
    
    st.info(f"Mode: **{complexity}**\n\nExplanations will be tailored to this level.")
    
    st.divider()
    
    # Clear Session
    if st.button("🧹 Clear Session", use_container_width=True):
        st.session_state.messages = []
        st.session_state.analysis_context = None
        st.session_state.current_file = None
        st.rerun()

# --- HELPER FUNCTIONS ---

def get_gemini_response(prompt, content=None, stream=True):
    """
    Interacts with Gemini 1.5 Pro.
    content: Can be an Image object or a Video File object.
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        
        # Construct the conversation history for context
        history = []
        for msg in st.session_state.messages:
            role = "user" if msg["role"] == "user" else "model"
            history.append({"role": role, "parts": [msg["content"]]})
            
        chat = model.start_chat(history=history)
        
        inputs = [prompt]
        if content:
            inputs.append(content)
            
        if stream:
            response = chat.send_message(inputs, stream=True)
            return response
        else:
            response = chat.send_message(inputs)
            return response.text
            
    except Exception as e:
        return f"Error: {str(e)}"

def process_video(uploaded_file):
    """Uploads video to Gemini API and waits for processing."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Uploading and processing video frame data..."):
        video_file = genai.upload_file(path=tmp_path)
        
        # Wait for processing
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = genai.get_file(video_file.name)
            
        if video_file.state.name == "FAILED":
            st.error("Video processing failed.")
            return None
            
    os.remove(tmp_path) # Clean up local
    return video_file

# --- MAIN LAYOUT ---

# Split screen: Input (Left) vs Output (Right)
col1, col2 = st.columns([1, 1.5], gap="medium")

# --- LEFT PANEL: INPUTS ---
with col1:
    st.subheader("1. Input Experiment Data")
    
    tab_cam, tab_up_img, tab_up_vid = st.tabs(["📸 Live Camera", "🖼️ Upload Image", "🎥 Upload Video"])
    
    input_content = None
    input_type = None

    # Camera Input
    with tab_cam:
        cam_image = st.camera_input("Take a photo of the experiment")
        if cam_image:
            input_content = Image.open(cam_image)
            input_type = "image"

    # Image Upload
    with tab_up_img:
        up_image = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
        if up_image:
            input_content = Image.open(up_image)
            st.image(input_content, caption="Uploaded Image", use_column_width=True)
            input_type = "image"

    # Video Upload
    with tab_up_vid:
        up_video = st.file_uploader("Upload Video", type=['mp4', 'mov', 'avi'])
        if up_video:
            st.video(up_video)
            if st.button("Process Video"):
                input_content = process_video(up_video)
                input_type = "video"
                st.session_state.current_file = input_content

    # Analysis Trigger
    if st.button("🔍 Analyze Experiment", type="primary", use_container_width=True):
        if not api_key:
            st.error("Please enter your API Key in the sidebar.")
        elif not input_content and not st.session_state.current_file:
            st.warning("Please provide an image or video first.")
        else:
            # Use stored file if video was already processed
            final_content = input_content if input_content else st.session_state.current_file
            
            # System Prompt for Initial Analysis
            system_prompt = f"""
            Act as an expert scientist and lab assistant. 
            Audience Level: {complexity}.
            
            Analyze the provided visual input of a science experiment. 
            Your response must follow this structure:
            1. **Equipment Identification**: List tools visible.
            2. **Process Analysis**: What is happening? (Reactions, phase changes, physical motion).
            3. **Safety Scan**: CRITICAL. Identify any unsafe setups, lack of PPE, or hazardous layouts. If safe, state "Setup appears safe."
            4. **Scientific Explanation**: Explain the concept deeply but suitable for a {complexity} level.
            5. **Prediction**: What will likely happen next?
            
            Be precise, educational, and friendly.
            """
            
            with col2:
                with st.spinner("Analyzing scientific data..."):
                    # Create a placeholder for streaming
                    response_placeholder = st.empty()
                    full_response = ""
                    
                    # Store input in session for follow-ups (Keep last image only for memory efficiency in this demo)
                    st.session_state.last_image = final_content if input_type == "image" else None
                    
                    try:
                        # Direct generation without chat history for the initial "vision" prompt
                        model = genai.GenerativeModel('gemini-1.5-pro-latest')
                        response = model.generate_content([system_prompt, final_content], stream=True)
                        
                        for chunk in response:
                            full_response += chunk.text
                            response_placeholder.markdown(full_response + "▌")
                        
                        response_placeholder.markdown(full_response)
                        
                        # Add to chat history
                        st.session_state.messages.append({"role": "user", "content": "Analyze this experiment visual."})
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                        st.session_state.analysis_context = full_response # Save for report generation
                        
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")

# --- RIGHT PANEL: RESULTS & CHAT ---
with col2:
    st.subheader("2. Assistant & Analysis")

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input for Follow-up
    if prompt := st.chat_input("Ask a follow-up question (e.g., 'Why did it turn blue?')"):
        if not api_key:
            st.error("API Key required.")
        else:
            # Display user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                # Contextual Prompt
                context_prompt = f"""
                You are a helpful lab assistant. 
                Current Context: {st.session_state.analysis_context}
                User Question: {prompt}
                Level: {complexity}
                Answer the question based on the visual context previously analyzed.
                """
                
                # If we have an image in memory, include it again for better context, otherwise text only
                content_to_send = st.session_state.get('last_image')
                
                resp_stream = get_gemini_response(context_prompt, content_to_send)
                
                if isinstance(resp_stream, str): # Error caught
                     message_placeholder.error(resp_stream)
                else:
                    for chunk in resp_stream:
                        full_response += chunk.text
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                    
                st.session_state.messages.append({"role": "assistant", "content": full_response})

    # --- AUTO LAB REPORT GENERATOR ---
    st.divider()
    if st.button("📝 Generate Official Lab Report", use_container_width=True):
        if not st.session_state.analysis_context:
            st.warning("Please analyze an experiment first to generate a report.")
        else:
            with st.spinner("Drafting structured lab report..."):
                report_prompt = f"""
                Based on the previous analysis:
                {st.session_state.analysis_context}
                
                Generate a formal Lab Report in Markdown format.
                Include these specific sections:
                1. **Title** (Creative and Accurate)
                2. **Objective**
                3. **Materials Identified**
                4. **Observations**
                5. **Theoretical Analysis** ({complexity} level)
                6. **Error Sources & Safety Notes**
                7. **Conclusion**
                
                Format it cleanly using Markdown headers.
                """
                
                # Generate report (Text-to-text is fine here as context contains the analysis)
                model = genai.GenerativeModel('gemini-1.5-pro-latest')
                report = model.generate_content(report_prompt).text
                
                st.markdown('<div class="report-box">', unsafe_allow_html=True)
                st.markdown(report)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Download Button
                st.download_button(
                    label="💾 Download Report as MD",
                    data=report,
                    file_name="experiment_report.md",
                    mime="text/markdown"
                )
              
