from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm
from torch import Tensor
from jaxtyping import Int, Float

from pipeline.utils.hook_utils import add_hooks

class ModelBase(ABC):
    def __init__(self, model_name_or_path: str):
        self.model_name_or_path = model_name_or_path
        self.model: AutoModelForCausalLM = self._load_model(model_name_or_path)
        self.tokenizer: AutoTokenizer = self._load_tokenizer(model_name_or_path)
        
        self.tokenize_instructions_fn = self.tokenize_conversations_fn
        self.eoi_toks = self._get_eoi_toks()

        self.model_block_modules = self._get_model_block_modules()
        self.model_attn_modules = self._get_attn_modules()
        self.model_mlp_modules = self._get_mlp_modules()

    def del_model(self):
        if hasattr(self, 'model') and self.model is not None:
            del self.model

    @abstractmethod
    def _load_model(self, model_name_or_path: str) -> AutoModelForCausalLM:
        pass

    @abstractmethod
    def _load_tokenizer(self, model_name_or_path: str) -> AutoTokenizer:
        pass

    @abstractmethod
    def _get_eoi_toks(self):
        pass

    @abstractmethod
    def _get_model_block_modules(self):
        pass

    @abstractmethod
    def _get_attn_modules(self):
        pass

    @abstractmethod
    def _get_mlp_modules(self):
        pass

    @abstractmethod
    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        pass

    @abstractmethod
    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff: float, layer: int):
        pass

    def get_conversation(self, message, chat_history):
        conversation = []
        for user, assistant in chat_history:
          conversation.extend(
								[
										{"role": "user", "content": user},
										{"role": "assistant", "content": assistant},
								]
						)

        conversation.append({"role": "user", "content": message})

        return conversation

    def tokenize_conversation(self, instructions, chat_histories=None):
        if chat_histories is None:
            chat_histories = [[] for _ in range(len(instructions))]
        conversations = [self.get_conversation(instruction, chat_history) for instruction, chat_history in zip(instructions, chat_histories)]
        toks = self.tokenizer.apply_chat_template(conversations, add_generation_prompt=True,padding=True,truncation=False, return_tensors="pt").to("cuda")
        return toks


    # This is just a wrapper that is easier to call
    def tokenize_conversations_fn(self, questions, prompts=None, first_responses=None):
        chat_histories = None
        if prompts and first_responses:
             chat_histories = [[[prompt, first_response]] for prompt, first_response in zip(prompts, first_responses)]

        return self.tokenize_conversation(questions, chat_histories)

    def _generate_single_answer(self, instructions, generation_config, fwd_pre_hooks, fwd_hooks, chat_histories):
        tokenized_instructions = self.tokenize_conversation(instructions=instructions, chat_histories=chat_histories)

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
                generation_toks = self.model.generate(
                    input_ids=tokenized_instructions.to(self.model.device),
                    generation_config=generation_config,
                )

                responses = []
                for instruction, generation in zip(instructions, generation_toks):
                    response = self.tokenizer.decode(generation, skip_special_tokens=True).strip()

                    instruction = instruction.strip()
			
                    # Remove the instruction from the response
                    last_message_index = response.rfind(instruction)
                    if last_message_index != -1:
                        response = response[last_message_index + len(instruction):].strip()

                    model_start_msgs = ['model\n', 'assistant\n']
                    for model_start_msg in model_start_msgs:
                      if response.startswith(model_start_msg):
                        response = response[len(model_start_msg):]

                    responses.append(response)


                chat_histories = [[[prompt, answer]] for prompt, answer in zip(instructions, responses)]

                return responses, chat_histories


    def generate_single_answer(self, dataset, fwd_pre_hooks=[], fwd_hooks=[], batch_size=2, max_new_tokens=300):
        """This function generates a response to a prompt."""
        generation_config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)
        generation_config.pad_token_id = self.tokenizer.pad_token_id

        completions = []
        instructions = [x['prompt'] for x in dataset]
        categories = [x['category'] for x in dataset]


        for i in tqdm(range(0, len(dataset), batch_size)):
                batch_responses, _ = self._generate_single_answer(instructions[i:i+batch_size], generation_config,
                                                                          fwd_pre_hooks, fwd_hooks, chat_histories=None)            

                for generation_idx, response, in enumerate(batch_responses):

                    completions.append({
                        'category': categories[i + generation_idx],
                        'prompt': instructions[i + generation_idx],
                        'last_response': response  # Keep the format consistent
                    })


        return completions


    def generate_completions(self, dataset, fwd_pre_hooks=[], fwd_hooks=[], batch_size=2, max_new_tokens=300):
        """This function generate a first response to a prompt and then a second response to a question."""
        generation_config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)
        generation_config.pad_token_id = self.tokenizer.pad_token_id

        completions = []
        instructions = [x['prompt'] for x in dataset]
        categories = [x['category'] for x in dataset]
        questions = [x['question'] for x in dataset]


        for i in tqdm(range(0, len(dataset), batch_size)):
                first_responses, chat_histories = self._generate_single_answer(instructions[i:i+batch_size], generation_config,
                                                                          fwd_pre_hooks, fwd_hooks, chat_histories=None)            

 
                last_responses, _ = self._generate_single_answer(questions[i:i+batch_size], generation_config,
                                                                          fwd_pre_hooks, fwd_hooks,
                                                                          chat_histories)        
                

                for generation_idx, (first_response, last_response, question) in enumerate(zip(first_responses,last_responses, questions[i:i+batch_size])):

                    completions.append({
                        'category': categories[i + generation_idx],
                        'prompt': instructions[i + generation_idx],
                        'question': question,
                        'first_response': first_response,
                        'last_response': last_response
                    })


        return completions
