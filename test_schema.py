import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

try:
    from backend.main import app
    print("Schema generation test...")
    schema = app.openapi()
    print("SUCCESS: OpenAPI schema generated.")
except Exception as e:
    import traceback
    print("FAILED: Schema generation error:")
    traceback.print_exc()
