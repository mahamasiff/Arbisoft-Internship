import os
from mcp.server.fastmcp import FastMCP
from google import genai

mcp = FastMCP("CatServer")

@mcp.resource("cats://care-guide")
def get_care_guide():
    file_path = os.path.join(os.path.dirname(__file__), "..", "week5", "catcare.txt")
    with open(file_path, "r") as f:
        return f.read()

@mcp.tool()
def calculate_cat_to_human_age(cat_years: float) -> str:

    if cat_years < 0:
        raise ValueError("Cat years cannot be negative.")
    if cat_years <= 1:
        human_equivalent = cat_years * 15
    elif cat_years <= 2:
        human_equivalent = 24
    else:
        human_equivalent = 24 + ((cat_years - 2) * 4)

    if cat_years < 1:
        stage = "Kitten"
    elif cat_years <= 6:
        stage = "Young Adult"
    elif cat_years <= 10:
        stage = "Mature"
    else:
        stage = "Senior"
        
    return f"A cat that has lived {cat_years} calendar years is approximately {int(human_equivalent)} in human years. Life Stage: {stage}."

if __name__ == "__main__":
    mcp.run()