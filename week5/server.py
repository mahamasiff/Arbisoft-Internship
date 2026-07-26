import os
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("CatServer")

@mcp.resource("cats://care-guide")
def get_care_guide():
    file_path = os.path.join(os.path.dirname(__file__), r"C:\Users\lenovo\Arbisoft-Internship\week4\cats.txt")
    with open(file_path, "r") as f:
        return f.read()