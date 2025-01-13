
import os
import csv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

"""
Required:
pip install accelerate
pip install tiktoken

Optional:
pip install flash-attn
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline,TRANSFORMERS_CACHE
import tqdm



print(TRANSFORMERS_CACHE)

#import shutil
#shutil.rmtree(TRANSFORMERS_CACHE)

torch.cuda.empty_cache()

torch.random.manual_seed(0)
model_id = "microsoft/Phi-3-small-8k-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    trust_remote_code=True
)
assert torch.cuda.is_available(), "This model needs a GPU to run ..."
device = torch.cuda.current_device()
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)



pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device
)


generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    #"temperature": 0.0,
    "do_sample": False,
}
def query(messages):
  """Sends a conversation history to the AI assistant and returns the answer.

  Args:
      messages (list): A list of dictionaries, each with "role" and "content" keys.

  Returns:
      str: The answer from the AI assistant.
  """

  try:
      output = pipe(messages, truncation=True, **generation_args)
  except Exception as e:
      print("Problems with message, reducing tokens to 7500 and prompt otimization")
      print(str(e))


      tokens = tokenizer.encode(messages[0]["content"]+"\n"+messages[1]["content"])


      tokens = tokens[:5000]

      messages = tokenizer.decode(tokens)


      output = pipe(messages, truncation=False, **generation_args)
  return output[0]['generated_text']


def promptLLM(dominant_narrative, sub_narrative,article_text):
  context="""
  Given a news article along with its dominant and sub-dominant narratives, generate a concise text (maximum 80 words) supporting these narratives without the need to explicitly mentioning them. The explanation should align with the language of the article and be direct and to the point. If the sub-dominant narrative is 'Other,' focus solely on supporting the dominant narrative. The response should be clear, succinct, and avoid unnecessary elaboration.
  """


  messages = [
      {"role": "system",
       "content": context },
      {"role": "user",
       "content": f'Dominant Narrative: {dominant_narrative} \n Sub-dominant Narrative: {sub_narrative} \n Article: {article_text}'}
  ]
  result = query(messages)
  words=result.split()
  words=words[:81]
  result=" ".join(words)
  return result

def getArticleTexts(apath=os.path.join("data","out_examples", "raw-articles"), npath=os.path.join(
    "data","out_examples", "narratives_gt.csv")):
  results=list()
  data=[]
  nar=dict()
  with open(npath, encoding="utf-8") as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
          #print(row)
          id,main_narr,sub_narr=row
          nar.update({id:(main_narr,sub_narr)})

  # Loop through the directory
  for filename in os.listdir(apath):
      if filename.endswith('.txt'):
          try:
            with open(os.path.join(apath,filename), 'r',encoding="utf-8") as file:
              #fid=filename.split(".")[0]
              fid=filename
              main_narr,sub_narr=nar[fid]
              text = file.read()
              data.append({"id":fid,"main_narr":main_narr,"sub_narr":sub_narr,"text":text})
          except Exception as e:
              print(str(e))
              print(filename)




  return(data)


import csv


def read_processed_files(tsv_file_path):
    # Initialize an empty list to store the values from the first column
    filenames = []

    try:
        # Open the TSV file
        with open(tsv_file_path, mode='r', encoding='utf-8') as file:
            # Create a CSV reader to read the TSV file
            reader = csv.reader(file, delimiter='\t')

            # Read through each row and append the value from the first column to the list
            for row in reader:
                if row:  # Check if the row is not empty
                    filenames.append(row[0])  # Append the first column value
    except FileNotFoundError:
        print(f"Error: The file {tsv_file_path} does not exist.")
        return []
    except Exception as e:
        print(f"Error: {e}")
        return []

    return filenames


def getBaseline(data,output_folder="subtask3_baselines",output_filename="baselines.tsv"):
  if not os.path.exists(output_folder):
    # Create the folder
    os.makedirs(output_folder)
    print(f'Folder "{output_folder}" created.')
  else:
    print(f'Folder "{output_folder}" already exists.')

  results=[]
  files_processed=read_processed_files(os.path.join(output_folder,output_filename))
  print(files_processed)
  for entry in data:
    print(entry)
    #print(entry["id"])
    #print(entry["main_narr"])
    #print(entry["text"])

    if(
            (entry["main_narr"]!="Other") and
            (not(entry["id"] in files_processed))
    ):
      expl=promptLLM(entry["main_narr"],entry["sub_narr"],entry["text"])
      #print(expl)
      expl=expl.strip()
      #line_result=f'{entry["id"]}\t{entry["main_narr"]}\t{entry["sub_narr"]}\t{expl}\n'
      line_result = f'{entry["id"]}\t{expl}\n'
      results.append(line_result)
      with open(os.path.join(output_folder,output_filename), 'a',encoding="utf-8") as f:
          f.write(f"{line_result}")





'''
USAGE EXAMPLE
'''


'''
for lang in ["PT","BG","EN","HI"]:
    print("Load Data for: " + lang)
    data=getArticleTexts(os.path.join("dev",lang,"subtask-3-documents"),os.path.join("dev",lang,"subtask-3-dominant-narratives.txt"))
    print("Getting the Baseline:")
    getBaseline(data=data,output_folder=os.path.join("subtask3_baseline",lang),output_filename="baseline_phi.txt")
'''