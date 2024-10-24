from tqdm import tqdm
import concurrent.futures
#python -m leandojo_project.proplogic_serv8.Evaluation.DFS.concurrent_example
import pickle
import time
from propositional_logic.random_gen.evaluation_access import *

class test_Class:


    def __init__(self, atribute):
        self.attribute = atribute
        self.dict_test = {}

    def test_fun(self, id):
        print(id)
        try:
            self.object.provide_tactic('','')
        except Exception as e:
            pass

        pickle.dump(id,open(f'/home/c5an/leandojo_project/atp_research/DFS/{id}.pkl','wb'))
    def search_test(self, time_sleep):
        with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
            list(tqdm(executor.map(self.eval, time_sleep), total=len(time_sleep)))

        print('finished one round')



    def eval(self, id: int) -> None:
        self.test_fun(id)

if __name__ == "__main__":
    ids = [3, 3]
    eval_object = test_Class(3)
    #eval_object.search_test(ids)
    object = SingleTheoremEval(5, int(0))

    pickle.dump(object,open(f'/home/c5an/leandojo_project/atp_research/DFS/temp/test.pkl','wb'))
    obj = pickle.load(open(f'/home/c5an/leandojo_project/atp_research/DFS/temp/test.pkl','rb'))

    print(obj.get_initial_prompt())