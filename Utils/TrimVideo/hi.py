from moviepy import VideoFileClip
import os

# --- Configuration ---
# IMPORTANT: Change this to the path of your actual 1-hour video file
input_video_path = 'front.mp4' 

# IMPORTANT: Change this to your desired output file path and name
output_video_path = 'trimmed.mp4' 

# Define the desired duration in minutes
trim_duration_minutes = 5

# Calculate the end time in seconds
trim_duration_seconds = trim_duration_minutes * 60 

# --- Video Trimming Logic ---
print(f"Attempting to trim video: {input_video_path}")
print(f"Desired trim duration: {trim_duration_minutes} minutes ({trim_duration_seconds} seconds)")
print(f"Output file will be: {output_video_path}")

# Check if the input file exists
if not os.path.exists(input_video_path):
    print(f"Error: Input video file not found at {input_video_path}")
else:
    try:
        # Load the video clip
        # Using 'with' statement ensures resources are properly closed
        with VideoFileClip(input_video_path) as clip:
            print(f"Video loaded successfully. Duration: {clip.duration:.2f} seconds")

            # Ensure the trim duration is not longer than the video itself
            if trim_duration_seconds > clip.duration:
                print(f"Warning: Trim duration ({trim_duration_seconds}s) is longer than the video duration ({clip.duration:.2f}s). Trimming to video end.")
                end_time = clip.duration
            else:
                end_time = trim_duration_seconds

            # Trim the clip from the beginning (0 seconds) up to the calculated end_time
            # subclip(start_time, end_time) - times can be in seconds, or (m, s), or (h, m, s)
            print(f"Trimming from 0 seconds to {end_time:.2f} seconds...")
            trimmed_clip = clip.subclip(0, end_time)

            # Write the trimmed clip to a new file
            # Specify codecs for compatibility (especially for MP4)
            print(f"Writing trimmed video to {output_video_path}...")
            # codec="libx264" is standard for MP4 video
            # audio_codec="aac" is standard for MP4 audio
            trimmed_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac") 

            print("\nVideo trimming completed successfully!")
            print(f"Output file saved as: {output_video_path}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Possible issues:")
        print("- The input video file path is incorrect.")
        print("- moviepy or its dependencies (like FFmpeg) are not correctly installed.")
        print("- The video file might be corrupted or in an unsupported format.")
        print("Please ensure you have FFmpeg installed or that moviepy installed it correctly upon first run.")