"""
Naive LLM
"""
from src.players.base_player import BasePlayer
from src.players.prompts import build_base_prompts, build_user_prompt
from src.players.llms import call_llm, parse_response
from src.utils.file_storage import JSONLStore


class NaiveLLMPlayer(BasePlayer):
    def __init__(self, player_id, model_name, starting_stack, output_dir=None):
        super().__init__(player_id, model_name, starting_stack, output_dir)
        if output_dir is not None:
            self.action_log = JSONLStore(f"{output_dir}/actions.jsonl")
        else:
            self.action_log = None

    def decide(self, state):
        # get base prompts
        llm_state, action_prompt, _ = build_base_prompts(state)

        # llm
        prompt = build_user_prompt(llm_state, extra_sections=None)
        response = call_llm(self.model_name, [
            {"role": "system", "content": action_prompt},
            {"role": "user", "content": prompt},
        ], json_mode=True)
        parsed, action = parse_response(response, llm_state["legal_actions"])

        # record
        if self.action_log is not None:
            self.action_log.append({
                "hand_index": state["hand_index"],
                "phase":      state["phase"],
                "action":     action,
                "intent":     (parsed or {}).get("intent"),
                "raw":        response,
            })
        return action

    def observe(self, final_state):
        pass
