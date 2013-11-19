
class Labels:
    ''' Class who structures de labels
    The goal of this class is to structure the division of the label vector
    in order to integrate it into the same structure.
    -------
    Author: Antonio
    Date: 29/10/2013
    ---Last modification:
    Author:
    Date:
    '''

    def __init__(self, y):
        ''' Instantiation of the class.
        INPUT:
            * y: labels obtained from the raw data. <type 'numpy.ndarray'>
        -------
        Author: Antonio
        Date: 29/10/2013
        '''
        self.matrix = y


    def __getitem__(self,key):
        #print key
        if key is "s":
            return self.matrix[:,:5]
        elif key is "w":
            return self.matrix[:,5:9]
        elif key is "k":
            return self.matrix[:,9:]
        else:
            return self.matrix[key]

    def get_block(self,i):
        ''' Fucntion which returns the part of the labels you want.
        There are 3 posibilities as 3 blocks the y has.
        Exists a four which is only the whole label vector.

        INPUT:
            * i: the part of the set you want or 0 if you want all. <type 'int'>
        OUTPUT:
            * y: labels considered in that moment. <type 'numpy.ndarray'>
        -------
        Author: Antonio
        Date: 29/10/2013
        '''
        if i==1:
            y = self.matrix[:,:5]
        elif i==2:
            y = self.matrix[:,5:9]
        elif i==3:
            y = self.matrix[:,9:]
        elif i==0:
            y = self.matrix
        # TODO
        # otherwise, error message
        return y

    
