import streamlit as st
from dataclasses import dataclass

@dataclass
class Content:
    def __init__(self, text: str = None, fig = None):
        self.text = text
        self.fig = fig

    def display(self):
        pass


