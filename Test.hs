{-# LANGUAGE ViewPatterns,TupleSections #-}
import System.IO.MMap
import qualified Data.ByteString as BS
import qualified Data.ByteString.Char8 as BS8
import Data.List
import Control.Monad
import System.Environment
import Control.Applicative
import Data.Trie
import Find
import Control.Monad.IO.Class

import Data.Enumerator hiding (map, filter, filterM, mapM)
import qualified Data.Enumerator.Binary as EB
import qualified Data.Enumerator.List as EL
import qualified Data.Enumerator as E

readCorpus :: IO BS.ByteString
readCorpus = mmapFileByteString "corpus" Nothing

-- prune before, and pre-calculate sizes.

-- main = do print . size . fromList . fmap (take 3 &&& return ()) . BS.tails =<< readCorpus
-- main = do t <- fromList . fmap (,()) . BS.tails <$> readCorpus
--           forever $ do q <- BS.getLine
--                        print q
--                        print $ lookupBy (return size) q t

printI' :: Iteratee BS.ByteString IO ()
printI' = do
    mx <- EL.head 
    case mx of
        Nothing   -> return ()
        Just frag -> do
            liftIO . BS.putStr $ frag
            printI'

catE :: Enumeratee FilePath BS.ByteString IO b
catE = EB.enumFile

main = run_ $ tree "data" $$ catE =$ printI'
    
readCorpusMMap s = do
    ((read -> n):_) <- return [show s] -- getArgs
    corpus <- readCorpus
    mapM print . filter (\l -> n == BS.length l) . fmap (BS.take n) $ BS.tails corpus
--           mapM print . fmap (BS.take 4) . sort . init $ BS.tails corpus
           -- print . fmap head . sort . init $ BS.tails corpus