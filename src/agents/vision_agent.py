import cv2
import numpy as np
from PIL import Image
from typing import Dict, List
from langchain.chat_models import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
import base64
from io import BytesIO