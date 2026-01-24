from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace 
from typing import TypedDict,Annotated
from dotenv import load_dotenv
import os

load_dotenv()

hf_token = os.getenv("HUGGINGFACE_API_KEY")
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="conversational", 
    max_new_tokens=512,
    huggingfacehub_api_token=hf_token
)

model=ChatHuggingFace(llm=llm)

class ReviewRequest(TypedDict):
    key_themes: Annotated[list[str], "Key themes discussed in the review"]
    review_text: Annotated[str, "The text of the review provided by the user"]
    sentiment: Annotated[str, "The sentiment of the review, either 'positive' or 'negative'"]
    pros: Annotated[list[str], "List of pros mentioned in the review"]
    cons: Annotated[str, "List of cons mentioned in the review"]

structured_model=model.with_structured_output(ReviewRequest)

result=structured_model.invoke("""iPhone 17 Review: The "Pro" Gap Has Finally Vanished Tested January 2026 | Price: $799

The Verdict: 4.5 / 5
For the first time in years, the standard iPhone is the one most people should buy. By finally bringing 120Hz ProMotion and a 48MP ultra-wide camera to the base model, Apple has made the "Pro" upgrade harder to justify than ever before.

At A Glance
PROS

ProMotion Display: Finally, 120Hz on the base model makes the UI feel buttery smooth.

Camera: The new 48MP Ultra-wide lens is a massive leap for landscape and macro shots.

Battery Life: The A19 chip efficiency is incredible; easily a 1.5-day phone.

Display Brightness: 3,000 nits peak outdoor brightness is blindingly good.

CONS

Charging Speed: Still capped at 27W wired charging while competitors push 100W+.

USB Speeds: The USB-C port is still limited to USB 2.0 data transfer speeds (slow).

Zoom: Lack of a dedicated telephoto lens is the only real reason to go Pro now.

Full Review
It has been four months since the iPhone 17 launched in September 2025, and after using it daily as my primary driver, one thing is clear: this is the "Super Cycle" upgrade we’ve been waiting for.

For the last three generations, the standard iPhone felt like a second-class citizen—stuck with a 60Hz screen while even $300 Android phones had 120Hz. That ends today. The iPhone 17 doesn't just catch up; it essentially cannibalizes its bigger brother, the iPhone 17 Pro.

Display: The Game Changer
The headline feature is the screen. The 6.3-inch Super Retina XDR display now supports ProMotion. If you’re coming from an iPhone 13, 14, or 15, the difference is jarring. Scrolling through Instagram, navigating iOS 19, and playing games feels instantly more responsive.

Apple also bumped the outdoor peak brightness to 3,000 nits. I tested this under direct sunlight in Mumbai, and it remained perfectly legible, actually outshining the Galaxy S25 in pure visibility.

Design: Refined, Not Redefined
Visually, not much has changed from the iPhone 16. You still get the aerospace-grade aluminum edges and the color-infused glass back. However, the Ceramic Shield 2 front glass claims 3x better scratch resistance. In my four months of using it without a screen protector, I have yet to see a single micro-scratch, which is a promising improvement over last year's softer glass.

Cameras: 48MP Everywhere
Apple has ditched the 12MP ultra-wide sensor. The iPhone 17 rocks a 48MP Fusion Main and a 48MP Ultra-wide.

The Main Lens: Crisp, natural, and the "Photonic Engine" seems less aggressive this year, leaving shadows looking like actual shadows rather than HDR paintings.

The Ultra-Wide: This is the star. Macro shots are now razor-sharp, and low-light landscape shots have significantly less noise.

The Selfie Cam: The front camera has been bumped to 18MP (up from 12MP). FaceTime calls look noticeably clearer, though it reveals pores you might prefer remained hidden.

Performance & A19 Chip
The A19 chip (built on the N3P process) is overkill for 99% of users. Benchmark scores show it outperforming the Snapdragon 8 Gen 5 in single-core speeds. More importantly, the 12GB of RAM (up from 8GB) means apps rarely close in the background. This RAM upgrade was necessary to support the new on-device Apple Intelligence features, which run snappy and fast.

The "Air" Elephant in the Room
We have to talk about the lineup. The iPhone 17 sits alongside the new ultra-slim iPhone 17 Air. While the Air turns heads with its 5mm thickness, the standard iPhone 17 destroys it in battery life and thermal performance. Unless you strictly prioritize style over substance, the standard 17 is the better phone.""")

# print(result)
print("Review Text:", result['review_text'])
print("Sentiment:", result['sentiment'])
print("Key Themes:", result['key_themes'])
print("Pros:", result['pros'])
print("Cons:", result['cons'])    
