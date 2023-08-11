import numpy as np
import torch
import torch.nn.functional as F
import utils
import models

def insert_into_queue(queue, element):
    if not queue:
        return [element]
    
    new_queue = queue + [element]
    for i in range(len(queue)):
        if element['log_prob'] < queue[i]['log_prob']:
            new_queue = queue[:i] + [element] + queue[i:]
            break

    return new_queue


def purge_queue(queue, max_moves):
    new_queue = []
    for element in queue:
        if len(element['moves']) >= max_moves:
            new_queue.append(element)
    return new_queue

if __name__ == '__main__':
    # Setup the inference model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.initialize_model()
    models.load_model(model)
    model.to(device)
    model.eval()

    # Get a cube to solve
    # nMoves = np.random.randint(1, 25)
    nMoves = 5
    orig_cube, orig_loc_embed, orig_color_embed, orig_is_solved = utils.receiveRandomCubes(np.array([nMoves]), device=device)

    # Initialize Queue
    queue = [dict(
        cube = orig_cube[0],
        loc_embed = orig_loc_embed,
        color_embed = orig_color_embed,
        moves = [],
        log_prob = 0
    )]
    best_solve = []

    # TODO: Add a value modifier to prioritize depth first

    # Try moves until solved
    count = 0
    while queue:
        count += 1

        if count % 100 == 0:
            max_moves = 0
            current_element = {
                'Log Prob': queue[-1]['log_prob'], 
                'N Moves': len(queue[-1]['moves'])
            }
            for element in queue:
                if len(element['moves']) > max_moves:
                    max_moves = len(element['moves'])

            print(f'Queue size: {len(queue)}  Max moves: {max_moves}, Current: {current_element}')
            print(value[0][0].item())

        step_dict = queue.pop()
        logits, value = model(
            step_dict['loc_embed'], 
            step_dict['color_embed']
        )
        logits = F.log_softmax(logits[0], 0).cpu().detach().numpy()
        logits += step_dict['log_prob']

        cubes, loc_embeds, color_embeds, is_solved = utils.getPossiblePositions(
            step_dict['cube'],
            device=device
        )

        if any(is_solved):
            idx = np.where(is_solved)[0][0]
            best_solve = step_dict['moves'] + [idx]
            queue = purge_queue(queue, len(best_solve) - 2)
            print(f'\n---A solve was found with {len(best_solve)} moves---\n')
            continue

        if (len(step_dict['moves']) < len(best_solve) - 3) or not best_solve:
            for i in range(len(logits)):
                element = dict(
                    cube = cubes[i],
                    loc_embed = loc_embeds[i][None, :],
                    color_embed = color_embeds[i][None, :],
                    moves = step_dict['moves'] + [i],
                    log_prob = logits[i]
                )

                queue = insert_into_queue(queue, element)
    

    print(best_solve)



    


        

