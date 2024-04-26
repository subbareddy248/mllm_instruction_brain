#!/bin/bash

BASE_DIRECTORY="NOT_SET"  # This is the path to the directory which contains the dataset folders
GPU_ID="NOT_SET" # This is the CUDA ID of the GPU to use for the experiments

# These are set to send tqdm updates to my Telegram account
TELEGRAM_BOT_TOKEN="6891982912:AAFUbsyhX3tQPsds2hGgTPwkTbvSzGpIRyo"
TELEGRAM_CHAT_ID="704338780"

if [ "$BASE_DIRECTORY" == "NOT_SET" ]; then
    echo "Please set the base directory path"
    exit 1
fi

if [ "$GPU_ID" == "NOT_SET" ]; then
    echo "Please set the CUDA GPU ID to use"
    exit 1
fi

tgme() {
    ./tgme -b "$TELEGRAM_BOT_TOKEN" -c "$TELEGRAM_CHAT_ID" "$@"
}

for subject in 1 2 5 7; do
    for prompt_number in {1..9}; do
        params="Subject $subject, Prompt $prompt_number"

        ####################################################

        model="Idefics_9B_Instruct"
        tgme "$model" "$params" "Starting"

        python3 idefics_9b_instruct.py \
            -b 2 -p "$prompt_number" -s "$subject" -g "$GPU_ID" \
            --telegram-bot-token "$TELEGRAM_BOT_TOKEN" --telegram-chat-id="$TELEGRAM_CHAT_ID" \
            -d "$BASE_DIRECTORY" && tgme "$model" "$params" "Success" || tgme "$model" "$params" "Failed"

        ####################################################

        model="Llava1.5_13B"
        tgme "$model" "$params" "Starting"

        python3 llava_1.5_13b.py \
            -b 2 -p "$prompt_number" -s "$subject" -g "$GPU_ID" \
            --telegram-bot-token "$TELEGRAM_BOT_TOKEN" --telegram-chat-id="$TELEGRAM_CHAT_ID" \
            -d "$BASE_DIRECTORY" && tgme "$model" "$params" "Success" || tgme "$model" "$params" "Failed"

        ####################################################

        model="InstructBlip_13b"
        tgme "$model" "$params" "Starting"

        python3 instruct_blip_13b.py \
            -b 2 -p "$prompt_number" -s "$subject" -g "$GPU_ID" \
            --telegram-bot-token "$TELEGRAM_BOT_TOKEN" --telegram-chat-id="$TELEGRAM_CHAT_ID" \
            -d "$BASE_DIRECTORY" && tgme "$model" "$params" "Success" || tgme "$model" "$params" "Failed"

        ####################################################

        model="mPlug_OWL_LLAMA_7B"
        tgme "$model" "$params" "Starting"

        python3 mplug_owl_llama_7b.py \
            -b 2 -p "$prompt_number" -s "$subject" -g "$GPU_ID" \
            --telegram-bot-token "$TELEGRAM_BOT_TOKEN" --telegram-chat-id="$TELEGRAM_CHAT_ID" \
            -d "$BASE_DIRECTORY" && tgme "$model" "$params" "Success" || tgme "$model" "$params" "Failed"

        ####################################################

        model="ViT_Huge"
        tgme "$model" "$params" "Starting"

        python3 vit_huge_patch14_224_in21k.py \
            -b 2 -p "$prompt_number" -s "$subject" -g "$GPU_ID" \
            --telegram-bot-token "$TELEGRAM_BOT_TOKEN" --telegram-chat-id="$TELEGRAM_CHAT_ID" \
            -d "$BASE_DIRECTORY" && tgme "$model" "$params" "Success" || tgme "$model" "$params" "Failed"

        ####################################################
    done
done

set -x
