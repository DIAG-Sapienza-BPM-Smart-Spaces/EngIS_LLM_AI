from transformers import LlamaTokenizer, MistralForCausalLM, StoppingCriteria, StoppingCriteriaList
import json
import re
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Load Model and Tokenizer
tokenizer = LlamaTokenizer.from_pretrained(
    "teknium/OpenHermes-2.5-Mistral-7B",
    trust_remote_code=True
)

model = MistralForCausalLM.from_pretrained(
    "teknium/OpenHermes-2.5-Mistral-7B",
    torch_dtype=torch.float16,
    device_map=device,
    load_in_8bit=False,
    load_in_4bit=True,
    use_flash_attention_2=False,
    low_cpu_mem_usage=True
)

# Step 2: Define Tools with Parameters
def calculator(expression):
    """A simple calculator tool."""
    try:
        result = eval(expression) 
        return f"The result is {result}."
    except Exception as e:
        return f"Error in calculation: {e}"


TOOLS = {
    "Calculator": {"function": calculator, "parameters": "expression (a mathematical expression to evaluate)"}
}

TOOL_NAMES = list(TOOLS.keys())

# Step 3: Define Custom Stopping Criteria
class StopOnObservation(StoppingCriteria):

    def __init__(self, target_sequence, prompt):
        self.target_sequence = target_sequence
        self.prompt=prompt
        self.last=None
        self.new_token=''

    def __call__(self, input_ids, scores, **kwargs):
        # Get the generated text as a string

        generated_text = tokenizer.decode(input_ids[0])
        if self.last:
            self.new_token = self.new_token+ generated_text.replace(self.last,'')

        # Check if the target sequence appears in the generated text
        if self.target_sequence in self.new_token:
            return True  # Stop generation
        self.last=generated_text
        return False  # Continue generation

    def __len__(self):
        return 1
      
    def __iter__(self):
        yield self

# Step 4: Define the ReActAgent Class
class ReActAgent:
    def __init__(self, model, tokenizer, tools):
        self.model = model
        self.tokenizer = tokenizer
        self.tools = tools

    def format_prompt(self, question):
        """Construct the exact prompt template with tool descriptions."""
        tools_description = "\n".join(
            [f"- {tool}: {desc['parameters']}" for tool, desc in self.tools.items()]
        )
        system_prompt = f"""Answer the following questions as best you can. You have access to the following tools:

{tools_description}

The way you use the tools is by specifying a json blob.
Specifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).

The only values that should be in the "action" field are: {TOOL_NAMES}

The $JSON_BLOB should only contain a SINGLE action and MUST be formatted as markdown, do NOT return a list of multiple actions. Here is an example of a valid $JSON_BLOB:

```
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
```
Make sure to have the $INPUT in the right format for the tool you are using, and do not put variable names as input if you can find the right values.

ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about one action to take. Only one action at a time in this format:
Action:
```
$JSON_BLOB
```
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times, you should take several steps when needed. The $JSON_BLOB must be formatted as markdown and only use a SINGLE action at a time.)

Youb must always end your output with the following format:

Thought: I now know the final answer
Final Answer: the final answer to the original input question

Now begin! Reminder to ALWAYS use the exact characters `Final Answer:` when you provide a definitive answer. 
Question: {question}""" 

        return system_prompt
         

    def generate_response(self, prompt, stop_criteria):
        """Generate a response with stopping criteria."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        outputs = self.model.generate(
            input_ids, 
            attention_mask=attention_mask,
            max_new_tokens=750, 
            temperature=0.8, 
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1, 
            do_sample=True, 
            stopping_criteria=stop_criteria
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def parse_json_blob(self, response):
        """Extract the JSON blob related to an action from the model's response."""
        # Locate the "Action:" keyword in the response
        match = re.search(r"Action:\s*(?:```)?\s*({.*?})\s*(?:```)?", response, re.DOTALL)
        if match:
            json_str = match.group(1).strip()  # Extract the JSON object string
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print(f"Malformed JSON: {json_str}")
                return None
        print("No valid action JSON found in response.")
        return None

    def interact(self, question):
        """Answer the question iteratively using the structured ReAct process."""
        # generate the initial prompt
        initial_prompt = self.format_prompt(question)

        while True:
            # generate response
            stop_criteria = StopOnObservation("Observation:", self.tokenizer)
            response = self.generate_response(initial_prompt, stop_criteria)

            # mask the initial prompt from the current generation
            new_content = response.replace(initial_prompt, "").strip()

            # process the new content line by line and find the final answer or action
            has_action = False
            final_answer = None
            for line in new_content.splitlines():
                if line.startswith("Action:"):
                    has_action = True
                elif line.startswith("Final Answer:"):
                    final_answer = line[len("Final Answer:"):].strip()

            # handle final answer immediately
            if final_answer:
                return final_answer

            # parse the action
            if has_action:
                try:
                    action_json = self.parse_json_blob(new_content)
                except json.JSONDecodeError:
                    return "I am unable to answer the question."
            else:
                return "I am unable to answer the question."

            if action_json is None:
                return "I am unable to answer the question."

            # execute the action
            tool_name = action_json.get("action")
            tool_input = action_json.get("action_input")

            if tool_name in self.tools:
                result = self.tools[tool_name]["function"](tool_input)
                new_content += f" {result}"
            else:
                new_content += "\nFinal Answer: I am unable to answer the question."
                return "I am unable to answer the question."

            # update the initial prompt to include the latest reasoning
            initial_prompt += f"\n{new_content}"


# Step 5: Interaction Loop
if __name__ == "__main__":
    agent = ReActAgent(model, tokenizer, TOOLS)
    print("ReAct Agent Initialized. Type 'exit' to quit.\n")

    while True:
        question = input("Your Question: ")
        if question.lower() == "exit":
            print("Exiting...")
            break

        answer = agent.interact(question)
        print(f"Agent's Answer: {answer}\n")