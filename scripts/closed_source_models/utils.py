import cv2
import os
import base64

def extract_frames(video_path, output_folder):
    filename = video_path.split("/")[-1].split(".")[0]
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    
    # Create output folder if it doesn't exist
    os.makedirs(os.path.join(output_folder, str(filename)), exist_ok=True)
    output_path = os.path.join(output_folder, str(filename))
    frame_count = 0
    
    # Loop through the video frames
    while True:
        # Read the next frame
        ret, frame = video_capture.read()
        
        if not ret:
            break
        
        # Resize the frame to the desired dimensions
        # resized_frame = cv2.resize(frame, (512, 512))
        
        # Construct output file path
        output_file = os.path.join(output_path, f"frame_{frame_count}.jpg")
        
        # Save the resized frame as an image file
        cv2.imwrite(output_file, frame)
        
        frame_count += 1
        
    # Release the video capture object
    video_capture.release()

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
def create_content(img_paths):
    content = []
    prompt_dict = {
        "type": "text",
        "text": "Please generate a detailed caption which describes all the given frames. Some of the frames may contain inconsistencies which are annotated with a green bounding box around them with the type/name of the inconsistency. Your generated caption should capture the details of the entire video, while also describing all the inconsistencies. Thus, properly look at all the given frames and the region marked by the green bounding boxes when describing the inconsistencies. Further, make sure to mention specific details about each of the inconsistencies, and mention the exact names of the inconsistencies from the marked green bounding box. Also, while describing the inconsistency please be as specific and detailed as possible, don't be vague/general about the inconsistency. Thus, the reader of the caption should perfectly understand what inconsistencies/anomalies are in the video, and what the video is about. Do not mention about the green bounding box in your response, it is only for you to identify the inconsistencies. Make sure to describe all the inconsistencies in your caption. Do not analyze the impact of the inconsistencies, you should only describe them. There is no need to mention when the inconsistencies start or end, just describe them."
    }
    content.append(prompt_dict)
    for img_path in img_paths:
        base64_image = encode_image(img_path)
        img_dict = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }
        content.append(img_dict)

    return content

def create_qa_prompt(img_paths, caption):
    content = []
    prompt_dict = {
    "type": "text",
    "text": """You are given a video input, which is generated by a state of the art AI algorithm. Thus, these videos look very natural and almost realistic, but they are actually synthetic and generated by an AI algorithm. The videos may have some inconsistencies or anomalies present in them, which are generally localized to only a specific location in the video as identified by the green bounding boxes in the video. The rest of the video appears completely natural or realistic. These specific inconsistency may last for only a few frames of the video, or may last for the entire video itself. The inconsistency or anomalies in the video are generally events and phenomenon which is not observed in real world and physical scenarios. You will also be given a caption as input which describes the video, along with the specific inconsistency present in the video. Based on the given video and caption input, your task is to formulate 3 diverse and misleading questions to test whether the multi-modal LLM model can correctly identify the options based on the inconsistencies present in the video or not. So, your generated questions should give the model few options to choose from to make its answer, and these options should be of high quality and also have misleading choices so that you can test deeper level of understanding of these multi-modal LLM models. Thus, the goal of these questions is to accurately assess the multi-modal LLM's ability to accurately identify the inconsistencies present in the video. Generate questions that comprise both interrogative and declarative sentences, utilizing different language styles, and provide an explanation for each. Your response should be presented as a list of dictionary strings with keys 'Q' for questions and 'A' for the answer. Follow these rules while generating question and answers:
    1. Do not provide answers in the question itself. For example, the ground-truth attribute or component which makes the video scene unusual should never be mentioned in the question itself.
    2. Ensure the questions are concrete and specific, and not vague or ambiguous.
    3. The questions should be formed based on your deep understanding of the video and the caption. Thus, properly read the caption and look at the given video to generate the questions.
    4. The questions should only pertain to the inconsistencies present in the video, and not about the video in general.
    5. You may also ask the model some misleading questions talking about non-existent inconsistencies in the video, to test the model's ability to differentiate between real and fake inconsistencies.
    6. Do not ask vague questions, and the answer should only contain one of the correct option mentioned in the question.
    7. In your question itself you must provide multiple choice options for the answer, and the answer should be one of the options provided in the question. Please ensure you provide option choices and their corresponding letters in the question itself.
    8. In your answer, only mention the correct option letter from the question. Make sure that the correct option letter is not always the same, and randomly shuffle the correct option letter for each question.
    9. You must only follow the below output format, and strictly must not output any other extra information or text.
    Your output format should be strictly as follows, without any additional information or text:
    [{\"Q\": 'first question A) <option1> B) <option2> C) <option3> D) <option4>', \"A\": 'Pick the correct option letter from A) B) C) D)'}, {\"Q\": 'second question A) <option1> B) <option2> C) <option3> D) <option4>', \"A\": 'Pick the correct option letter from A) B) C) D)'}, ... }]
    Given below is the caption input which describes the given video along with the specific inconsistency present in the video.
    The caption is: """ + caption
    }
    content.append(prompt_dict)
    for img_path in img_paths:
        base64_image = encode_image(img_path)
        img_dict = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }
        content.append(img_dict)

    return content

def create_vqa_eval_prompt(img_paths, question):
    content = []
    prompt_dict = {
    "type": "text",
    "text": f"{question}"
    }
    content.append(prompt_dict)
    for img_path in img_paths:
        base64_image = encode_image(img_path)
        img_dict = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }
        content.append(img_dict)

    return content