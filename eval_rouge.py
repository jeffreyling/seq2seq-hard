from pyrouge import Rouge155
import os
import sys

# python eval_rouge.py /path/to/pred/summaries /path/to/gold/summaries
args = sys.argv[1:]
pred = args[0]
gold = args[1]

# These are created by bash script
SYSTEM_DIR = '/scratch/tmp_PRED'
MODEL_DIR = '/scratch/tmp_GOLD'

def write_lines(filename, outdir, name, gold):
  if not gold:
    os.mkdir(outdir + '/asdf')

  with open(filename, 'r') as f:
    for i,line in enumerate(f):
      cur_name = "{}-{}".format(name, i)
      if gold:
        d = outdir + '/' + cur_name
        os.mkdir(d)
        outfilename = "{}/{}.1.gold".format(d, cur_name)
        with open(outfilename, 'w') as out_f:
          out_f.write(line)
      else:
        with open("{}/asdf/{}.asdf.system".format(outdir, cur_name), 'w') as out_f:
          out_f.write(line)

  print 'Wrote lines for', filename, 'to', outdir

# Make temporary files
write_lines(pred, SYSTEM_DIR, 'cnn', False)
write_lines(gold, MODEL_DIR, 'cnn', True)

# r = Rouge155()
# r.system_dir = SYSTEM_DIR
# r.model_dir = MODEL_DIR
# r.system_filename_pattern = 'cnn.(\d+).txt'
# r.model_filename_pattern = 'cnn.#ID#.txt'

# output = r.convert_and_evaluate()
# print(output)
# # output_dict = r.output_to_dict(output)
