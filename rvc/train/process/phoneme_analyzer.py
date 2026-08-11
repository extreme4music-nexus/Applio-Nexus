import os
import whisper
import traceback
import nltk

def analyze_phoneme_coverage(dataset_folder):
    """
    Transcribes audio clips using a lightweight Whisper model 
    and evaluates phonetic diversity via standard G2P phoneme distribution maps.
    """
    try:
        if not dataset_folder or not os.path.exists(dataset_folder):
            return 0.0

        # Scan for valid training audio formats
        files = [f for f in os.listdir(dataset_folder) if f.endswith((".wav", ".mp3", ".flac"))]
        if not files:
            return 0.0

        # --- FORCE EXPLICIT NLTK RESOURCE VALIDATION ---
        print(f"[Smart Dataset Suite] Validating phonetic dependency maps...")
        try:
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)
            nltk.download('cmudict', quiet=True)
        except Exception as nltk_err:
            print(f"[Smart Dataset Suite] NLTK downpass skipped or offline: {str(nltk_err)}")
        # -----------------------------------------------

        print(f"[Smart Dataset Suite] Loading Whisper 'tiny' model into memory...")
        model = whisper.load_model("tiny")
        
        print(f"[Smart Dataset Suite] Initializing G2P Phonetic Engine...")
        from g2p_en import G2p
        g2p = G2p()
        
        phoneme_set = set()
        
        print(f"[Smart Dataset Suite] Analyzing {len(files)} files for phonetic density...")
        for filename in files:
            path = os.path.join(dataset_folder, filename)
            try:
                result = model.transcribe(path)
                text = result.get("text", "").strip()
                
                if not text:
                    continue
                
                # Convert text to standard phoneme list (e.g., ["H", "AE", "L", "OW"])
                phonemes = g2p(text)
                for ph in phonemes:
                    # Clean out spaces, punctuation markings, and stress numbers (like 'AA1' -> 'AA')
                    cleaned_ph = ''.join([c for c in ph if c.isalpha()])
                    if cleaned_ph:
                        phoneme_set.add(cleaned_ph)
            except Exception as file_err:
                print(f"[Smart Dataset Suite] Warning processing {filename}: {str(file_err)}")
                continue

        # The standard CMUBased clean English phoneme core inventory consists of 39 base phonemes
        target_phoneme_count = 39.0
        coverage_percentage = (len(phoneme_set) / target_phoneme_count) * 100.0
        
        print(f"[Smart Dataset Suite] Found {len(phoneme_set)} distinct phonemes in this batch.")
        return min(coverage_percentage, 100.0)

    except Exception:
        traceback.print_exc()
        return 0.0