# Training

batch_size = 4
if samples//batch_size < samples/batch_size:
  batches = (samples//batch_size)+1
else:
  batches = samples//batch_size
for e in range(100):
  index = list(range(samples))
  np.random.shuffle(index)
  accuracy = 0
  for batch in range(batches):
    bs = batch*batch_size
    be = bs+batch_size
    selected_indexes = index[bs:be]
    x,y = create_dataset(videos, encoded_labels, selected_indexes)
    # print(x.shape)
    # print(y.shape)
    results = classifier.train_on_batch(x,y)
    print('\r',batch,'/',batches,' : ',results[0],results[1],end='')
    accuracy += results[1]
    if batch%100 == 0:
      classifier.save_weights("classifier.h5")
  print('\r> ',e,', Accuracy = ',accuracy/batches)
  classifier.save_weights("classifier.h5")
