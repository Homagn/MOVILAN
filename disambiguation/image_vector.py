from img2vec_pytorch import Img2Vec
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import glob

def similarity(rel_loc = "", query = "downloads/test.jpg", support = "beanbag"):
	# Initialize Img2Vec with GPU
	#img2vec = Img2Vec(cuda=True) #we dont have cuda
	img2vec = Img2Vec()

	# Read in an image
	#img = Image.open('test3.jpg')
	img = Image.open(query)
	# Get a vector from img2vec, returned as a torch FloatTensor
	#vec = img2vec.get_vec(img, tensor=True)
	vec1 = img2vec.get_vec(img)
	#print("Got image vector ",vec1)
	# Or submit a list
	#vectors = img2vec.get_vec(list_of_PIL_images)
	Sim = 0.0
	for f in glob.glob(rel_loc+"downloads/*"):
		#print(f)
		if support in f:
			#print("filename ",f)
			#img = Image.open('test4.jpg')
			img = Image.open(f)
			vec2 = img2vec.get_vec(img)
			#print("Got image vector ",vec2)

			similarity = cosine_similarity(vec1.reshape((1, -1)), vec2.reshape((1, -1)))[0][0]
			#print("Similarity is ",similarity)
			Sim+=similarity
	#print("Average similarity ",Sim/5.0)
	return Sim/5.0


if __name__ == '__main__':
	similarity()