import numpy as np

def main():

    movies = np.loadtxt("hw8_movies.txt", dtype=str)
    ids = np.loadtxt("hw8_ids.txt", dtype=str)
    probR_init = np.loadtxt("hw8_probR_init.txt", dtype=float)
    probZ_init = np.loadtxt("hw8_probZ_init.txt", dtype=float)
    ratings = np.loadtxt("hw8_ratings.txt", dtype=str)

    # Compute mean popularity rating of each movie
    mean_ratings = computeMeanRatings(ratings)
    # printRatingInfo(movies, mean_ratings)

    print(EM(ratings, probR_init, probZ_init, 1))

def EM(ratings, probR_init, probZ_init, iterations):
    assert (np.shape(ratings)[0] == 258)
    assert (np.shape(ratings)[1] == 76)
    num_people = np.shape(ratings)[0]
    num_movies = np.shape(ratings)[1]
    num_cat = 4

    ratings = np.array(ratings)
    inferred_student_cat = np.zeros((num_people, num_cat))

    # Initialize probabilities
    Z = np.array(probZ_init)
    R = np.array(probR_init)

    # Estimate
    for istudent in range(num_people):
        for icat in range(num_cat):

            # Compute product of all P(Rj=rj | Z=i)
            numerator = Z[icat]
            for imovie in range(num_movies):
                # if ratings[istudent, j] == '1' or ratings[istudent, j] == '0':
                if ratings[istudent, imovie] == '1':
                    numerator *= R[imovie, icat] # P(Rj = 1 | Z=icat)
                elif ratings[istudent, imovie] == '0':
                    numerator *= (1 - R[imovie, icat]) # P (Rj = 0 | Z=icat)

            denominator = 0
            for icat1 in range(num_cat):
                temp = Z[icat1]
                for imovie1 in range(num_movies):
                    if ratings[istudent, imovie1] == '1':
                        temp *= R[imovie1, icat1]
                    elif ratings[istudent, imovie1] == '0':
                        temp *= (1- R[imovie1, icat1])
                denominator += temp

            inferred_student_cat[istudent, icat] = numerator/denominator
    
    return inferred_student_cat




    # Maximize


def computeMeanRatings(ratings):

    # Convert to array so we can manipulate more easily
    ratings_np = np.array(ratings)

    # Initialize return array, one element for each mean rating
    mean_ratings = np.zeros(np.shape(ratings_np)[1])
    total_times_watched = np.zeros(np.shape(ratings_np[1]))

    for iperson in range(np.shape(ratings_np)[0]):
        for imovie in range(np.shape(ratings_np)[1]):

            if ratings_np[iperson, imovie] == '1':
                mean_ratings[imovie] += 1
                total_times_watched[imovie] += 1
            elif ratings_np[iperson, imovie] == '0':
                total_times_watched[imovie] += 1

    for imovie in range(len(mean_ratings)):
        mean_ratings[imovie] = mean_ratings[imovie]/total_times_watched[imovie]

    return mean_ratings

def printRatingInfo(movies, mean_ratings):

    # Zips ratings and movies into a list, sorts it, then separates it back out
    mean_ratings, movies_sorted = (list(t) for t in zip(*sorted(zip(mean_ratings, movies))))
    print("***          Ratings          ***")
    for i in range(len(mean_ratings)):
        print("%s. %s: %s"%(i, movies_sorted[i], mean_ratings[i]))

if __name__ == "__main__":
    main()