from .data import *
from .state_tracking_ctx import *
from .theorem_encoding import *

import re

#python -m leandojo_project.proplogic_serv8.Evaluation.evluation_access
#python
tactic_label_format = re.compile(r"state_(\d+)_tactic_(\d+):")

back_track_label_format = re.compile(r"no solution, return to state (\d+) \[that leads to state (\d+)\]")

class SingleTheoremEval:

    def __init__(self, num_vars: int, encoding: int):
        self.num_vars = num_vars
        prop = decode_prop(encoding, num_vars)
        self.prop = prop
        self.ctx = StateTrackingCtx(num_vars, prop)

    def get_initial_prompt(self) -> str:
        """ Get the initial prompt for the theorem.
        Must be called before any other method.
        """
        assert self.ctx.get_cur_state_num() == 0, "Must be at initial state"
        return f"state_0:\n" + self.ctx.get_cur_tactic_state()
    
    def get_current_state_with_label(self) -> str:
        return f"state_{self.ctx.get_cur_state_num()}:\n" + \
            self.ctx.get_cur_tactic_state()

    def do_back_track(self, back_tactic: str) -> Tuple[str, str]:
        """
        Backtrack by providing the exact tactic 
        no solution, return to state (\d+) \[that leads to state (\d+)\]
        """
        state_with_label_before = self.get_current_state_with_label()

        matches = back_track_label_format.match(back_tactic)
        if matches is None:
            raise ValueError("Invalid back track label", back_tactic)
        back_track_to_state = int(matches.group(1))
        back_track_from_state = int(matches.group(2))
        if self.ctx.get_cur_state_num() != back_track_from_state:
            raise ValueError("Backtrack from the wrong state ", back_track_from_state, "expecting", self.ctx.get_cur_state_num())
        self.ctx.set_cur_state_num(back_track_to_state)

        state_with_label_after = self.get_current_state_with_label()
        return (state_with_label_before, state_with_label_after)



    def provide_tactic(self, label: str, tactic: str) -> Tuple[str,str]:
        """
        Provide a tactic for the current state.
        Returns one or more states.
        """
        state_with_label_before = self.get_current_state_with_label()

        label = label.strip()
        tactic = tactic.strip()
        match = tactic_label_format.match(label)
        if match is None:
            raise ValueError("Invalid tactic label", label)
        state_num = int(match.group(1))
        tactic_num = int(match.group(2))
        if self.ctx.get_cur_state_num() != state_num:
            raise ValueError("Must provide tactic from the right state num", state_num, "expecting", self.ctx.get_cur_state_num())
        self.ctx.push_state(self.ctx.get_cur_state_text() + "\n" + self.ctx.current_indent + tactic)
        result = []
        while True:
            result.append(f"state_{self.ctx.get_cur_state_num()}:\n")
            if (tactic.startswith("have") and tactic.endswith("by")) \
                or (tactic.startswith("case") and tactic.endswith("=>")):
                self.ctx.increment_indent()
            current_tactic_state = self.ctx.get_cur_tactic_state()
            result.append(current_tactic_state + "\n")
            if current_tactic_state != "no goals":
                break
            else:
                if self.ctx.current_indent == "  ":
                    result.append("proof is complete")
                    break
                else:
                    self.ctx.decrement_indent()
                    self.ctx.push_state(self.ctx.get_cur_state_text())
        state_with_label_after =  "".join(result).strip()
        return (state_with_label_before, state_with_label_after)

    def get_current_lean_proof(self):
        return self.ctx.get_cur_state_text()


if __name__ == "__main__":
    sample_eval = SingleTheoremEval(5, 346960918443446424220011675436)
    print(sample_eval)
    tactic = ["state_0_tactic_0:", "have thm1 : True := by","state_1_tactic_0:", "apply True.intro", "state_3_tactic_0:", "apply True.intro"]
    print("\n".join(tactic))
    print(sample_eval.get_initial_prompt())
    for i in range(0, len(tactic), 2):
        print("\n".join(tactic[i:i+2]))
        print(sample_eval.provide_tactic(tactic[i], tactic[i+1]))
    # print(sample_eval.provide_tactic(tactic[0], tactic[1]))
    print(sample_eval.get_current_lean_proof())

