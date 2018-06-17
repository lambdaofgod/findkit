

class Index:

    def find_similar(self, query_object, n_returned):
        """
        Perform nearest neighbor query on index

        Parameters
        ----------
        query_object : numpy array (of shape (dimensionality,))
            object for which nearest neighbors are found

        n_returned : int
            number of returned most similar objects


        Returns
        -------
        indices : iterable of int
            indices of nearest neighbors

        distances : iterable of float
            distances between query_object and nearest neighbors

        """
        raise NotImplementedError()