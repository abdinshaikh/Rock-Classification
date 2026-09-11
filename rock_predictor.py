from google.colab import files
from tensorflow.keras.preprocessing import image
from IPython.display import display, HTML
import tensorflow as tf
import numpy as np
import base64
import io


def predict_rock(model, classes):

    # 1. Upload image

    uploaded = files.upload()

    if not uploaded:
        return

    img_path = next(iter(uploaded))

    # 2. Load image

    img = image.load_img(
        img_path,
        target_size=(256, 256),
        color_mode="rgb"
    )

    img_array = image.img_to_array(img)
    input_array = tf.expand_dims(img_array, axis=0)

    # 3. Prediction

    predictions = model.predict(
        input_array,
        verbose=0
    )[0]

    # 4. Top 3 predictions
    top_indices = np.argsort(predictions)[::-1][:3]

    top_classes = [classes[i] for i in top_indices]

    top_confidences = [predictions[i] * 100 for i in top_indices]

    predicted_class = top_classes[0]
    confidence = top_confidences[0]

    # 5. Confidence / uncertainty

    second_confidence = top_confidences[1]
    confidence_gap = confidence - second_confidence

    if confidence >= 80 and confidence_gap >= 20:
        level = "HIGH CONFIDENCE"
        level_class = "high"

    elif confidence >= 50 and confidence_gap >= 10:
        level = "MODERATE CONFIDENCE"
        level_class = "moderate"

    else:
        level = "LOW CONFIDENCE"
        level_class = "low"

    # 6. Convert image to Base64

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")

    img_base64 = base64.b64encode(
        buffer.getvalue()
    ).decode()

    # 7. Top-3 prediction rows

    prediction_rows = ""

    for rank, (name, conf) in enumerate(
        zip(top_classes, top_confidences),
        start=1
    ):

        prediction_rows += f"""
        <div class="prediction">

            <div class="rank">{rank}</div>

            <div class="prediction-name">
                {name}
            </div>

            <div class="small-bar-bg">
                <div class="small-bar"
                     style="width:{conf:.2f}%;">
                </div>
            </div>

            <div class="prediction-value">
                {conf:.2f}%
            </div>

        </div>
        """

    # 8. Warning

    warning = ""

    if confidence_gap < 10:

        warning = """
        <div class="warning">
            <span>
                Top two predictions are very close.
                Prediction should be treated with caution.
            </span>
        </div>
        """

    # 9. HTML UI

    html = f"""

    <style>

        /* Main Card */

        .rock-card {{
            width: 820px;
            max-width: 95%;
            margin: 10px auto;
            background: #ffffff;
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 5px 20px rgba(0,0,0,0.12);
            font-family: Arial, Helvetica, sans-serif;
            color: #1f2937 !important;
        }}


        /* Header */

        .rock-header {{
            background: #1f2937;
            color: #ffffff !important;
            text-align: center;
            padding: 14px 20px;
        }}

        .rock-header h1 {{
            margin: 0;
            color: #ffffff !important;
            font-size: 21px;
            font-weight: 700;
        }}

        .rock-header p {{
            margin: 4px 0 0 0;
            color: #d1d5db !important;
            font-size: 12px;
        }}


        /* Main Content */

        .rock-content {{
            display: grid;
            grid-template-columns: 45% 55%;
            min-height: 410px;
        }}


        /* Image Section */

        .image-panel {{
            background: #f3f4f6;
            padding: 22px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }}

        .image-panel h3 {{
            margin: 0 0 12px 0;
            color: #374151 !important;
            font-size: 14px;
            font-weight: 600;
        }}

        .rock-image {{
            width: 285px;
            height: 285px;
            object-fit: cover;
            border-radius: 12px;
            background: #ffffff;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}


        /* Result Section */

        .result-panel {{
            padding: 28px 30px;
            background: #ffffff;
            color: #1f2937 !important;
        }}

        .result-label {{
            color: #6b7280 !important;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 1px;
            text-transform: uppercase;
        }}

        .predicted-rock {{
            color: #111827 !important;
            font-size: 30px;
            font-weight: 700;
            margin-top: 3px;
            margin-bottom: 8px;
        }}


        /* Confidence */

        .confidence-text {{
            color: #374151 !important;
            font-size: 14px;
            margin-bottom: 7px;
        }}

        .confidence-text strong {{
            color: #111827 !important;
        }}

        .confidence-bg {{
            width: 100%;
            height: 10px;
            background: #e5e7eb;
            border-radius: 20px;
            overflow: hidden;
            margin-bottom: 9px;
        }}

        .confidence-fill {{
            height: 100%;
            width: {confidence:.2f}%;
            background: #2563eb;
            border-radius: 20px;
        }}


        /* Confidence Label */

        .confidence-level {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 10px;
            font-weight: 700;
            margin-bottom: 22px;
        }}

        .high {{
            background: #dcfce7;
            color: #166534 !important;
        }}

        .moderate {{
            background: #fef3c7;
            color: #92400e !important;
        }}

        .low {{
            background: #fee2e2;
            color: #991b1b !important;
        }}


        /* Top 3 */

        .top-title {{
            color: #111827 !important;
            font-size: 14px;
            font-weight: 700;
            margin-bottom: 14px;
        }}

        .prediction {{
            display: grid;
            grid-template-columns: 22px 80px 1fr 55px;
            align-items: center;
            gap: 7px;
            margin-bottom: 13px;
        }}

        .rank {{
            color: #6b7280 !important;
            font-size: 12px;
            font-weight: 700;
        }}

        .prediction-name {{
            color: #374151 !important;
            font-size: 12px;
        }}

        .small-bar-bg {{
            height: 8px;
            background: #e5e7eb;
            border-radius: 10px;
            overflow: hidden;
        }}

        .small-bar {{
            height: 100%;
            background: #2563eb;
            border-radius: 10px;
        }}

        .prediction-value {{
            color: #374151 !important;
            font-size: 11px;
            font-weight: 600;
            text-align: right;
        }}


        /* Warning */

        .warning {{
            display: flex;
            gap: 8px;
            align-items: flex-start;
            margin-top: 18px;
            padding: 10px 12px;
            background: #fff7ed;
            border: 1px solid #fed7aa;
            border-radius: 8px;
            color: #9a3412 !important;
            font-size: 11px;
            line-height: 1.4;
        }}


        /* Responsive */

        @media (max-width: 700px) {{

            .rock-content {{
                grid-template-columns: 1fr;
            }}

            .rock-image {{
                width: 230px;
                height: 230px;
            }}

        }}

    </style>


    <div class="rock-card">

        <!-- HEADER -->

        <div class="rock-header">

            <h1>
                ROCK CLASSIFICATION
            </h1>

            <p>
                Rock Identification System using VGG16 Fine-Tuning
            </p>

        </div>


        <!-- CONTENT -->

        <div class="rock-content">


            <!-- IMAGE -->

            <div class="image-panel">

                <h3>
                    Uploaded Rock Image
                </h3>

                <img
                    class="rock-image"
                    src="data:image/jpeg;base64,{img_base64}"
                />

            </div>


            <!-- RESULTS -->

            <div class="result-panel">

                <div class="result-label">
                    Predicted Rock
                </div>

                <div class="predicted-rock">
                    {predicted_class}
                </div>


                <div class="confidence-text">

                    Confidence:
                    <strong>
                        {confidence:.2f}%
                    </strong>

                </div>


                <div class="confidence-bg">

                    <div class="confidence-fill">
                    </div>

                </div>


                <div class="confidence-level {level_class}">
                    {level}
                </div>


                <div class="top-title">
                    Top 3 Predictions
                </div>


                {prediction_rows}


                {warning}

            </div>

        </div>

    </div>
    """

    display(HTML(html))

    # Console output

    print("ROCK CLASSIFICATION RESULT")

    print(f"Predicted Rock : {predicted_class}")
    print(f"Confidence     : {confidence:.2f}%")
    print(f"Confidence Gap : {confidence_gap:.2f}%")
    print(f"Assessment     : {level}")

    print("\nTop 3 Predictions:")

    for rank, (name, conf) in enumerate(
        zip(top_classes, top_confidences),
        start=1
    ):
        print(f"{rank}. {name:<12} {conf:.2f}%")
