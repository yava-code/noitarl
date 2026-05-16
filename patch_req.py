with open("requirements.txt", "r") as f:
    content = f.read()

import re
content = re.sub(r'<<<<<<< HEAD.*?=======\n', '', content, flags=re.DOTALL)
content = content.replace('>>>>>>> origin/main\n', '')

# add missing test reqs
content += """
# Development
black>=23.0.0
flake8>=6.0.0
mypy>=1.5.0
pytest>=7.4.0
pytest-cov>=4.1.0
pygetwindow
"""

with open("requirements.txt", "w") as f:
    f.write(content)
