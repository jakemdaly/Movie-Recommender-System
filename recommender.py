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

    R, Z, inferred_student_cat, LL_over_time = EM(ratings, probR_init, probZ_init, 10)
    my_movierating_vec = np.array(ratings)[107]
    personalRecommendations(my_movierating_vec, R, Z, inferred_student_cat, movies)


def EM(ratings, probR_init, probZ_init, iterations):
    assert (np.shape(ratings)[0] == 258)
    assert (np.shape(ratings)[1] == 76)
    num_people = np.shape(ratings)[0]
    num_movies = np.shape(ratings)[1]
    num_cat = 4
    LL_over_time = []
    ratings = np.array(ratings)
    inferred_student_cat = np.zeros((num_people, num_cat))

    # Initialize probabilities
    Z = np.array(probZ_init)
    R = np.array(probR_init)

    for iter in range(iterations):

        student_likelihoods = []

        # Estimate
        for istudent in range(num_people):
            likelihood = 0
            for icat in range(num_cat):

                # Compute product of all P(Rj=rj | Z=i)
                numerator = Z[icat]
                for imovie in range(num_movies):
                    # if ratings[istudent, j] == '1' or ratings[istudent, j] == '0':
                    if ratings[istudent, imovie] == '1':
                        numerator *= R[imovie, icat] # P(Rj = 1 | Z=icat)
                    elif ratings[istudent, imovie] == '0':
                        numerator *= (1 - R[imovie, icat]) # P (Rj = 0 | Z=icat)

                likelihood += numerator

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

            student_likelihoods.append(likelihood)

        LL = 0
        for istudent4 in range(num_people):
            LL += (1/num_people)*np.log(student_likelihoods[istudent4])
        LL_over_time.append(LL)
        print("Iteration %s LL: %s"%(iter, LL))

        # Maximize
        Z_new = np.zeros(num_cat)
        for icat2 in range(num_cat):
            for istudent1 in range(num_people):

                Z_new[icat2] += (1/num_people)*inferred_student_cat[istudent1, icat2]

        R_new = np.zeros(np.shape(R))

        for icat3 in range(num_cat):
            for imovie2 in range(num_movies):

                numerator = 0
                for istudent2 in range(num_people):

                    if ratings[istudent2, imovie2] == '0' or ratings[istudent2, imovie2] == '1':
                        numerator += inferred_student_cat[istudent2, icat3] * int(ratings[istudent2, imovie2])
                    elif ratings[istudent2, imovie2] == '?':
                        numerator += inferred_student_cat[istudent2, icat3] * R[imovie2, icat3]
                    else:
                        print("Error in maximizing P(Rj=1|Z=i)")
                        assert(False)

                denominator = 0
                for istudent3 in range(num_people):
                    denominator += inferred_student_cat[istudent3,icat3]

                R_new[imovie2, icat3] = numerator/denominator

        Z = Z_new
        R = R_new

    return R, Z, inferred_student_cat, LL_over_time

def personalRecommendations(my_movierating_vec, R, Z, inferred_student_cat, movies):
    num_cat = 4

    probs = []
    for imovie in range(len(my_movierating_vec)):

        if my_movierating_vec[imovie] == '?':

            temp = 0
            for i in range(num_cat):
                temp += inferred_student_cat[107, i] * R[imovie, i]
            probs.append([imovie, temp])

    unseen_movies = []
    probability_of_liking = []
    for iunseen_movie in probs:
        unseen_movies.append(movies[iunseen_movie[0]])
        probability_of_liking.append(iunseen_movie[1])

    movies, p_liking = (list(t) for t in zip(*sorted(zip(unseen_movies, probability_of_liking))))
    print("***      PERSONAL RECOMMENDATIONS        ***")
    for j in range(len(movies)):
        print(f"Movie: {movies[j]}, Rating: {p_liking[j]}")


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