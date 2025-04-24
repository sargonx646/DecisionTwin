#!/bin/bash
     # Install agentiq from NVIDIA's private repository
     if [ -n "$NVIDIA_API_KEY" ]; then
         pip install agentiq==1.0.0 langchain==0.2.16 langchain-community==0.2.15 langchain-core==0.2.38 faiss-cpu==1.8.0 --extra-index-url https://build.nvidia.com || echo "Failed to install agentiq; proceeding with Grok-3-Beta only."
     else
         echo "NVIDIA_API_KEY not set; skipping agentiq installation."
     fi
     # Start Streamlit
     streamlit run app.py --server.port=8501 --server.address=0.0.0.0
