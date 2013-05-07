This is a tutorial for how to use the Hidden Markov Model (HMM) library in Haskell.  In this tutorial, we will use the Data.HMM package to find genes in the second chromosome of Vitis vinifera: the wine grape vine.  Finding genes is a common task in bioinformatics where HMMs have seen a lot of use.

The basic procedure goes in three steps.  First, we create an HMM to model the chromosome.  We do this by running the Baum-Welch training algorithm on all the DNA.  Second, we create an HMM to model transcription factor binding sites.  This is where genes are located.  Finally, we use Viterbi's algorithm to determine which HMM best models the DNA at a given location in the chromosome.  If it's the first, this is probably not the start of a gene.  If it's the second, it probably it.

Unfortunately, it's beyond the scope of this tutorial to go into the math of HMMs and how they work.  Instead, we will focus on how to use them in practice.  And like all good Haskell tutorials, this page is actually a literate Haskell program, so you can simply cut and paste it into your favorite text editor to run it.

The code
-----

Before we do anything else, we must import the HMM libraries

>import Data.HMM

   import Data.HMM.HMMFile

We'll also need some other libraries for this program

>import Control.Monad
>import Data.Array

-- >import Data.Number.LogFloat

>import System.IO

Now, let's create our first HMM.  The HMM datatype is:

data HMM stateType eventType = HMM { states :: [stateType]
                                   , events :: [eventType]
                                   , initProbs :: (stateType -> Prob)
                                   , transMatrix :: (stateType -> stateType -> Prob)
                                   , outMatrix :: (stateType -> eventType -> Prob)
                                   }

Notice that states and events can be any type supported by Haskell.  In this example, we will be using both integers and strings for the states, and characters for the events.  DNA is composed of 4 base pairs that get repeated over and over: adenine (A), guanine (G), cytosine (C), and thymine (T), so "AGCT" will be the list of our events.

To create an HMM corresponding to the diagram shown below:  

>hmm1 = HMM { states=[1,2]
>           , events=['A','G','C','T']
>           , initProbs = ip
>           , transMatrix = tm
>           , outMatrix = om
>           }
>
>ip s
>    | s == 1  = 0.1
>    | s == 2  = 0.9
>
>tm s1 s2
>    | s1==1 && s2==1    = 0.9
>    | s1==1 && s2==2    = 0.1
>    | s1==2 && s2==1    = 0.5
>    | s1==2 && s2==2    = 0.5
>
>om s e
>    | s==1 && e=='A'    = 0.4
>    | s==1 && e=='G'    = 0.1
>    | s==1 && e=='C'    = 0.1
>    | s==1 && e=='T'    = 0.4
>    | s==2 && e=='A'    = 0.1
>    | s==2 && e=='G'    = 0.4
>    | s==2 && e=='C'    = 0.4
>    | s==2 && e=='T'    = 0.1

While creating HMMs manually is straightforward, we will typically want to use one of the built in HMMs.  This simplest way to do this is the function simpleHMM:

>hmm2 = simpleHMM [1,2] "AGCT"

hmm2 is an HMM with the same states and events as hmm1, but all the transition probabilities are distributed in an unknown manner.  This is okay, however, because we will normally want to train our HMM using Baum Welch to determine those parameters automatically.

Another simple way to create an HMM is by creating a straight Markov Model with the simpleMM command.  (Note the absence of an "H")  Below, hmm3 is a 3rd order markov model for DNA:

>hmm3 = simpleMM "AGCT" 3
            
Now, how do we train our model?  The standard algorithm is called Baum-Welch.  To illustrate the process, we'll create a short array of DNA, then call three iterations of baumWelch on it.

>dnaArray = listArray (1,20) "AAAAGGGGCTCTCTCCAACC"
>hmm4 = baumWelch hmm3 dnaArray 3

We use arrays instead of lists because this gives us better performance when we start passing large training data to Baum-Welch.  Doing three iterations is completely arbitrary.  Baum-Welch is guaranteed to converge, but there is no way of knowing how long that will take.

Now, let's train our HMM on an entire chromosome.  We will have to load it from a file like this:

>loadDNAArray len = do
>    dna <- readFile "dna/winegrape-chromosone2"
>    let dnaArray = listArray (1,len) $ filter isBP dna
>    return dnaArray
>    where
>          isBP x = if x `elem` "AGCT" -- This filters out the "N" base pair, meaning we don't know what's at that location
>                      then True
>                      else False
>
>createDNAhmm file len hmm = do
>    dna <- loadDNAArray len
>    let hmm' = baumWelch hmm dna 10
>    putStrLn $ show hmm'
>    saveHMM file hmm'
>    return hmm'

The loadDNAArray function simply loads the DNA from a file into an array, and the createDNAhmm function actually calls the baumWelch algorithm.  This function can take a while on long inputs---and DNA is a long input!---so we also pass a file parameter for it to save our HMM when it's done for later use.  So let's create our HMM:

>hmmDNA = createDNAhmm "trained.hmm" 50000 hmm3

This call takes almost a full day on my laptop.  Luckily, you don't have to repeat it.  The Data.HMM.HMMFile module allows us to write our HMMs to disk and retrieve them later.  Simply call loadHMM:

>hmmDNA_file = loadHMM "trained.hmm" :: IO (HMM String Char)

NOTE: Whenever you use loadHMM, you must specify the type of the resulting HMM.  loadHMM relies on the built-in "read" function, and this cannot work unless you specify the type!

Great!  Now, we have a fully trained HMM for our chromosome.  Our next step is to train another HMM on the transcription factor binding sites.  There are many advanced ways to do this (e.g. Profile HMMs), but that's beyond the scope of this tutorial.  We're simply going to download a list of TF binding sites, concatenate them, then train our HMM on them.  This won't be as effective, but saves us from taking an unnecessary tangent.

>createTFhmm file hmm = do
>    x <- strTF
>    let hmm' = baumWelch hmm (listArray (1,length x) x) 10
>    putStrLn $ show hmm'
>    saveHMM file hmm'
>    return hmm'
>    where 
>          strTF = liftM (concat . map ((++) "")) loadTF
>          loadTF = liftM (filter isValidTF) $ (liftM lines) $ readFile "TFBindingSites"
>          isValidTF str = (length str > 0) && (not $ elemChecker "#(/)[]|N" str)
>
>elemChecker :: (Eq a) => [a] -> [a] -> Bool
>elemChecker elemList list 
>    | elemList == []  = False
>    | otherwise       = if (head elemList) `elem` list
>                           then True
>                           else elemChecker (tail elemList) list

Now, let's create our transcription factor HMM:

>hmmTF = createTFhmm "TF-3.hmm" $ simpleMM "AGCT" 3

Or if you're in a hurry, just load it from a file:

>hmmTF_file = loadHMM "TF-3.hmm" :: IO (HMM String Char)

So now we have 2 HMMs, how are we going to use them?  We'll combine the two HMMs into a single HMM, then use Viterbi's algorithm to determine which HMM best characterizes our DNA at a given point.  If it's hmmDNA, then we do not have a TF binding site at that location, but if it's hmmTF, then we probably do.

The Data.HMM library provides another convenient function for combining HMMs, hmmJoin.  It adds transitions from every state in the first HMM to every state in the second, and vice versa, using the "joinParam" to determine the relative probability of making that transition.  This is the simplest way to combine to HMMs.  If you want more control over how they get combined, you can implement your own version.
    
>findGenes len joinParam hout = do
>    hmmTF <- loadHMM "hmm/TF-3.hmm" :: IO (HMM String Char)
>    hmmDNA <- loadHMM "hmm/autowinegrape-1000-3.hmm"  :: IO (HMM String Char)
>    let hmm' = seq hmmDNA $ seq hmmTF $ hmmJoin hmmTF hmmDNA joinParam
>    dna <- loadDNAArray len
>--     putStrLn $ show hmm'
>    hPutStrLn hout ("len="++show len++",joinParam="++show joinParam++" -> "++(show $ concat $ map (show . fst) $ viterbi hmm' dna))
>
>main = do
>    hout <- openFile "BioResults" WriteMode
>    mapM_ (\len -> mapM_ (\jp -> findGenes len jp hout) [0.5,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.6]) [50000]
>    hClose hout

Finally, our main function runs findGenes with several different joinParams.  These act as thresholds for finding where the genes actually occur.