# 1. Import the library

from inference_sdk import InferenceHTTPClient



# 2. Connect to your workspace

client = InferenceHTTPClient(

  api_url="https://serverless.roboflow.com",

  api_key="zs2Fpdr6GYFftsJ7kQws"

)



# 3. Run your workflow on an image

result = client.run_workflow(

  workspace_name="dylans-workspace-gchst",

  workflow_id="general-segmentation-api-3",

  images={

    "image": r"C:\Users\dylan\Documents\wake-waste-ai\dataset\images\train\000031.jpg"

  },

  parameters={

    "classes": "Cigarette, Battery, Shoe, Glass bottle, Paper cup, Plastic straw, "
        "Drink can, Garbage bag, Pop tab, Paper bag, Styrofoam piece, Plastic lid, "
        "Plastic film, Broken glass, Other carton, Drink carton, Corrugated carton, "
        "Normal paper, Food waste, Crisp packet, Glass jar, Meal carton, Egg carton, "
        "Food Can, Aerosol, Aluminium foil, Paper straw, Squeezable tube, Glass cup, "
        "Metal lid, Spread tub, Foam cup, Scrap metal, Magazine paper, Wrapping paper, "
        "Toilet tube, Tupperware, Polypropylene bag, Pizza box, Aluminium blister pack, "
        "Carded blister pack, Clear plastic bottle, Disposable food container, "
        "Disposable plastic cup, Foam food container, Metal bottle cap, Other plastic, "
        "Other plastic bottle, Other plastic container, Other plastic cup, Other plastic wrapper, "
        "Plastic bottle cap, Plastic glooves, Plastic utensils, Rope - strings, Rope & strings, "
        "Single-use carrier bag, Six pack rings, Tissues, Unlabeled litter"

  },

  use_cache=True  

)



# 4. Filter and display only the necessary results

# Navigate into the specific nested layout of your Roboflow response

workflow_output = result[0] if isinstance(result, list) else result

predictions_data = workflow_output.get("predictions", {})

detected_items = predictions_data.get("predictions", [])



print("=" * 40)

print(f"TOTAL TRASH COUNT: {workflow_output.get('trash_count', 0)}")

print("=" * 40)



for item in detected_items:

    print(f"Class:      {item.get('class')}")

    print(f"Confidence: {item.get('confidence'):.2%}")

    print(f"Box Size:   Width: {item.get('width')}, Height: {item.get('height')}")

    print(f"Center Pos: X: {item.get('x')}, Y: {item.get('y')}")

    print("-" * 40)