ALL_PROMPTS = {
    # Image Captioning
    1: "In this task, you will look at the image and briefly describe the image.",
    2: "What is the caption of the image?",
    3: "Generate some text to describe the image.",
    4: "Look at image and tell me what is the content.",
    5: "In this task, you are given an image and you will need to generate some text to describe it.",

    # VQA
    6: "What is in this image?",
    7: "Are there any people in this image? If yes, describe them.",
    8: "What is the foreground of the image? What is in the background?",
    9: "What primary colors dominate the scene?",

    # Visual Relationship
    10: "What kind of interaction is happening between the animate and inanimate objects here?",
    11: "What objects are being used by the largest animal in this image?",
    12: "Which animal is interacting with the water body?",
    13: "Locate the food item closest to the cooking appliance.",
    14: "Determine the position of the bird in the forest.",

    # Commonsense Reasoning
    15: "Identify if the object on the left is larger than the object on the right.",
    16: "Based on the image, decide if it is daytime.",
    17: "What time of the day does the image likely represent?",
    18: "From the image, can you conclude if it is currently raining?",
    19: "What type of environment is shown in the image?",

    # Image Understanding
    20: "Describe the most dominant color in the image.",
    21: "Is the setting indoors or outdoors?",
    22: "List any food items visible.",
    23: "Which object is closest to the camera?",
    24: "Are there any trees in the image?",
    25: "How many animals are there in the image?",
    26: "Is the focus of the camera on the foreground or background?",
    27: "Is there any object that could help in cooking?",
    28: "Does the image feel chaotic or peaceful?",
    29: "Identify any sports equipment displayed in the image.",

    # Temporal Ordering
    30: "Given the context of the image, predict the immediate next action.",

    # Grounded Generation
    31: "Describe the top-left quarter of the image.",
    32: "Highlight the area that shows a natural outdoor scene.",
    33: "What are the regions containing animals?",
    34: "Highlight the regions where text is visible in the image.",
    35: "Highlight the region containing the animal on the left, which is sitting.",
    36: "Generate the referring expression for the tree in the top left corner of the image.",
    37: "Determine the type of primary object in the bottom centre of the image.",

    # Grounded Matching
    38: "Check if the two clothing items in the image are identical.",
    39: "Which of the four corners of the image shows sports equipment?",
    40: "Which of the four corners of the image shows the person wearing the red shirt?",
    41: "Do the two animals depicted belong to the same genus?",
    42: "Which of the four quarters of the image contains a vehicle?",

    # Region Understanding
    43: "Estimate the area covered by the vehicle displayed in the image.",
    44: "Which of the four corners of the image overlaps the most with the region of the animal in the photo?",
    45: "Select a region that does not overlap with the water body in the image.",
    46: "Which of the four corners of the image overlaps the least with the region containing the sports equipment?",
    47: "Estimate the area of the image which is not covered by the grass.",

    # Image-Text Matching
    48: '''Chose the option that best matches what is happening in the image
        a. Sports activity
        b. Indoor activity
        c. Animals roaming around
        d. None of the above
    ''',
    49: '''Chose the option that best matches what is depicted in the image
        a. People having a meal
        b. Furniture decoration in a house
        c. Vehicles parked on a road
        d. None of the above
    ''',
    50: "Decide if the image contains an answer to the question, “Is there any wildlife in the image?”",
    51: "Decide if the image contains an answer to the question, “Is there any vehicles present in the image”?",

    # Text Legibility
    52: "Decide if the text in the image is readable.",
    53: "Determine the clarity of the text.",
    54: "Confirm the legibility of the text based on its appearance.",
    55: "Check the text for readability issues.",
    56: "Is the text present in the image unclear to read?",

    # Visual Text Extraction
    57: "Extract the text from the image.",
    58: "Identify and extract all the text from the image.",
    59: "Transcribe any and all the text from the image.",
    60: "Gather all the textual content from the image.",
    61: "Read and extract the text from the picture.",

    # Disaster Type Classification
    62: "Identify the type of disaster depicted in the image.",
    63: "Classify the disaster shown in the picture.",
    64: "Assess the nature of the disaster occurring in the image.",
    65: "Evaluate the type of disaster shown.",
    66: "Specify the disaster type based on the image.",

    # # Visual Dialogue
    # 67: "Based on the image, what would you ask the person in the scene?",
    # 68: "Given the scene, what would be your next question?",
    # 69: "Generate a dialogue that could be occurring in the image.",
    # 70: "Given the context of the image, what question arises?",
    # 71: "Craft a question that relates to the image.",
}
