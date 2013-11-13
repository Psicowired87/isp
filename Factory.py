
class Factory:
    ''' Class that generalizes the vctVectorizer mehod.
    -------
    Author: Antonio
    Date: 29/10/2013
    ---Last modification:
    Author:
    Date:
    '''

    
    def create(self,method,params): 
        '''
        INPUTS:
            * params: dictionary integrate the inputs
                * measure: String with the type of measure
                * aMaxN: max size of ngram to get as features
        -------
        Author: Antonio
        Date: 29/10/2013
        ---Last modification:
        Author:
        Date:
        '''
       
        comp  = method()
        
        for param in params:
            setattr(comp, param,params[param])
            
        return comp
        
#    def create_vect(self,method,params): 
 #       '''
 #       INPUTS:
  #          * params: dictionary integrate the inputs
   #             * measure: String with the type of measure
    #            * aMaxN: max size of ngram to get as features
     #   -------
#        Author: Antonio
 #       Date: 29/10/2013
  #      ---Last modification:
   #     Author:
    #    Date:
     #   '''
      # 
       # vct  = method()
        
       #for param in params:
        #    setattr(vct, param,params[param])
#            
 #       return vct
        
  #  def create_model(self,method,params): 
   #     '''
    #    INPUTS:
     #       * params: dictionary integrate the inputs
      #          * measure: String with the type of measure
       #         * aMaxN: max size of ngram to get as features
#        -------
 #       Author: Antonio
  #      Date: 29/10/2013
   #     ---Last modification:
    #    Author:
     #   Date:
      #  '''
#       
 #       model = Model()
  #      
   #     for param in params:
    #        setattr(model, param,params[param])
     #   model.clf
      #  return vct
    
        
    
