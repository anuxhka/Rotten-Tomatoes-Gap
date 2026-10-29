"""ASGI entry point for a Vercel Python Function.

Vercel imports ``app``; Streamlit executes the dashboard script per session.
Keep the dashboard in the project root so its data and asset paths resolve.
"""

from pathlib import Path

import streamlit as st


app = st.App(Path(__file__).resolve().parents[1] / "app.py")
