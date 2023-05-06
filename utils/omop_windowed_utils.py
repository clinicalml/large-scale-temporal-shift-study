import sys
import json
import pandas as pd
import numpy as np
import sparse
import os
import gc
import time

from os.path import dirname, abspath, join
from omop_learn.utils.data_utils import to_unixtime, from_unixtime
from omop_learn.data.common import ConceptTokenizer

def process_line(line):
    '''
    Process line of data into a json object with visit dates and times
    @param line: str
    @return: json object
    '''
    example = json.loads(line)
    dates = example['dates']
    unix_times = to_unixtime(dates)
    example['unix_times'] = unix_times

    # make sure visits are sorted by date
    sorted_visits = [v for d, v in sorted(zip(example['unix_times'], example['visits']))]
    example['visits'] = sorted_visits
    example['unix_times'] = sorted(example['unix_times'])
    example['dates'] = sorted(example['dates'])

    return example

def build_3d_sparse_feature_matrix(tokenizer,
                                   data_folder,
                                   logger):
    '''
    Build sparse feature matrix
    Writes intermediate chunks to disk to run faster
    @param tokenizer: ConceptTokenizer
    @param data_folder: str, folder containing data.json
    @param logger: logger, for INFO messages
    @return: 1. 3d sparse feature matrix
             2. list of times corresponding to dim 1 in matrix
             3. list of person_ids corresponding to dim 0 in matrix
    '''
    chunk_size = 10000
    final_chunk_file = data_folder + 'sparse_matrix_components_chunks_finished.txt'
    if os.path.exists(final_chunk_file):
        with open(final_chunk_file, 'r') as f:
            final_chunk_text = f.read()
        num_people = int(final_chunk_text.split(' ')[1])
    else:
        # Build 3d sparse feature matrix
        # After every chunk of samples, save to disk and start building these again from an empty list
        concepts   = []
        times      = []
        persons    = []
        times_set  = set()
        person_ids = []

        # Build in smaller chunks of samples and then extend overall lists above
        small_chunk_size       = 100
        small_chunk_concepts   = []
        small_chunk_times      = []
        small_chunk_persons    = []
        small_chunk_times_set  = set()
        small_chunk_person_ids = []

        start_time = time.time()
        # Process the json data file by line
        # A line constitutes an entire person worth of data
        with open(data_folder + 'data.json', 'r') as json_fh:
            for person_id, line in enumerate(json_fh):

                # This is a person
                example = process_line(line)
                small_chunk_person_ids.append(example['person_id'])

                # These are the visits, which can have many concepts each
                for i, v in enumerate(example['unix_times']):

                    # This is the number of concepts in this visit
                    visit_concept_num = len(example['visits'][i])

                    # Extend lists by the number of concepts in this visit
                    small_chunk_concepts.extend(example['visits'][i])
                    small_chunk_times.extend([v]*visit_concept_num)
                    small_chunk_persons.extend([person_id]*visit_concept_num)

                    # Make a time set for use in mapping later
                    small_chunk_times_set.add(v)
                del example

                if person_id % small_chunk_size == small_chunk_size - 1:
                    # add these small chunks of samples to larger chunks
                    concepts.extend(small_chunk_concepts)
                    times.extend(small_chunk_times)
                    persons.extend(small_chunk_persons)
                    times_set = times_set.union(small_chunk_times_set)
                    person_ids.extend(small_chunk_person_ids)
                    small_chunk_concepts   = []
                    small_chunk_times      = []
                    small_chunk_persons    = []
                    small_chunk_times_set  = set()
                    small_chunk_person_ids = []
                    gc.collect()
                    logger.info('Processed ' + str(person_id + 1) + ' people in sparse feature matrix creation in '
                                + str(time.time() - start_time) + ' seconds')

                if person_id % chunk_size == chunk_size - 1:
                    # save chunk to disk
                    save_start_time = time.time()
                    concepts_mapped = tokenizer.concepts_to_ids(concepts)
                    with open(data_folder + 'sparse_matrix_components_chunk' + str(person_id + 1) + '.json', 'w') as f:
                        json_contents = {'concepts_mapped': concepts_mapped,
                                         'times'          : times,
                                         'times_set'      : list(times_set),
                                         'persons'        : persons,
                                         'person_ids'     : person_ids}
                        json.dump(json_contents, f)
                    del concepts_mapped
                    concepts   = []
                    times      = []
                    persons    = []
                    times_set  = set()
                    person_ids = []
                    logger.info('Saved chunk ' + str(person_id + 1) + ' to disk in '
                                + str(time.time() - save_start_time) + ' seconds')

        num_people = person_id + 1
        if len(small_chunk_person_ids) > 0:
            concepts.extend(small_chunk_concepts)
            times.extend(small_chunk_times)
            persons.extend(small_chunk_persons)
            times_set = times_set.union(small_chunk_times_set)
            person_ids.extend(small_chunk_person_ids)
            del small_chunk_concepts
            del small_chunk_times
            del small_chunk_persons
            del small_chunk_times_set
            del small_chunk_person_ids
        
        if len(person_ids) > 0:
            # save remaining samples after last complete chunk to disk
            save_start_time = time.time()
            concepts_mapped = tokenizer.concepts_to_ids(concepts)
            with open(data_folder + 'sparse_matrix_components_chunk' + str(num_people) + '.json', 'w') as f:
                json_contents = {'concepts_mapped': concepts_mapped,
                                 'times'          : times,
                                 'times_set'      : list(times_set),
                                 'persons'        : persons,
                                 'person_ids'     : person_ids}
                json.dump(json_contents, f)
                del json_contents
            del concepts_mapped
            del concepts
            del times
            del persons
            del times_set
            del person_ids
            gc.collect()
            logger.info('Saved last chunk ' + str(num_people) + ' to disk in '
                        + str(time.time() - save_start_time) + ' seconds')
        else:
            logger.info('Last chunk already contained all people. No additional chunk to save to disk.')
        with open(data_folder + 'sparse_matrix_components_chunks_finished.txt', 'w') as f:
            f.write('Saved ' + str(num_people) + ' people in chunks to disk for sparse feature matrix creation')

        logger.info('Finished processing all ' + str(num_people) + ' people to chunks in sparse feature matrix creation in '
                    + str(time.time() - start_time) + ' seconds')
    
    # Read all chunks back to create sparse matrix
    start_time          = time.time()
    concepts_mapped_list_over_chunks = []
    times_list_over_chunks           = []
    all_times_set                    = set()
    persons_list_over_chunks         = []
    person_ids_list_over_chunks      = []
    chunk_end                        = chunk_size
    while chunk_end < num_people:
        # load chunk
        with open(data_folder + 'sparse_matrix_components_chunk' + str(chunk_end) + '.json', 'r') as f:
            json_contents = json.load(f)
        concepts_mapped_list_over_chunks.append(np.array(json_contents['concepts_mapped']))
        times_list_over_chunks.append(json_contents['times'])
        all_times_set = all_times_set.union(set(json_contents['times_set']))
        persons_list_over_chunks.append(np.array(json_contents['persons']))
        person_ids_list_over_chunks.append(np.array(json_contents['person_ids']))
        logger.info('Loaded chunks of ' + str(chunk_end) + ' people in ' 
                    + str(time.time() - start_time) + ' seconds')
        chunk_end += chunk_size
        
    if num_people % chunk_size != 0:
        # load last chunk
        with open(data_folder + 'sparse_matrix_components_chunk' + str(num_people) + '.json', 'r') as f:
            json_contents = json.load(f)
        concepts_mapped_list_over_chunks.append(np.array(json_contents['concepts_mapped']))
        times_list_over_chunks.append(json_contents['times'])
        all_times_set = all_times_set.union(set(json_contents['times_set']))
        persons_list_over_chunks.append(np.array(json_contents['persons']))
        person_ids_list_over_chunks.append(np.array(json_contents['person_ids']))
    logger.info('Loaded all chunks of ' + str(num_people) + ' people in '
                + str(time.time() - start_time) + ' seconds')
    
    # Now make a dict of our times
    start_time = time.time()
    all_times_list = sorted(list(all_times_set))
    del all_times_set
    all_times_map  = {visit_time: i for i, visit_time in enumerate(all_times_list)}

    # Equivalent to ConceptTokenizer concepts_to_ids
    all_times_mapped_list_over_chunks = []
    for chunk_times in times_list_over_chunks:
        chunk_times_mapped = np.empty(len(chunk_times), dtype=int)
        for idx in range(len(chunk_times)):
            chunk_times_mapped[idx] = all_times_map[chunk_times[idx]]
        all_times_mapped_list_over_chunks.append(chunk_times_mapped)
    del times_list_over_chunks
    gc.collect()
    logger.info('Mapped times to indices in ' + str(time.time() - start_time) + ' seconds')
    
    # Now concatenate all lists over chunks
    start_time = time.time()
    all_concepts_mapped = np.concatenate(concepts_mapped_list_over_chunks)
    all_times_mapped    = np.concatenate(all_times_mapped_list_over_chunks)
    all_persons         = np.concatenate(persons_list_over_chunks)
    all_person_ids      = np.concatenate(person_ids_list_over_chunks)
    del concepts_mapped_list_over_chunks
    del all_times_mapped_list_over_chunks
    del persons_list_over_chunks
    del person_ids_list_over_chunks
    gc.collect()
    logger.info('Concatenated all chunks in ' + str(time.time() - start_time) + ' seconds')

    # Build 3D sparse matrix representation of the data
    # persons x times x concepts
    start_time      = time.time()
    feature_matrix  = sparse.COO(
      [all_persons, all_times_mapped, all_concepts_mapped], 1, 
      shape=(len(set(all_persons)), 
             len(all_times_map), 
             len(tokenizer.concept_map))
    )
    del all_persons
    del all_times_mapped
    del all_concepts_mapped
    gc.collect()
    logger.info('Built 3d sparse matrix in ' + str(time.time() - start_time) + ' seconds')
    
    return feature_matrix, all_times_list, all_person_ids