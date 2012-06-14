{-# LANGUAGE ViewPatterns,TupleSections #-}
import System.IO.MMap
import qualified Data.ByteString as BS
import Data.List
import Control.Monad
import System.Environment
import Control.Applicative
import Data.Trie

readCorpus :: IO BS.ByteString
readCorpus = mmapFileByteString "corpus" Nothing

-- prune before, and pre-calculate sizes.

-- main = do print . size . fromList . fmap (take 3 &&& return ()) . BS.tails =<< readCorpus
main = do t <- fromList . fmap (,()) . BS.tails <$> readCorpus
          forever $ do q <- BS.getLine
                       print q
                       print $ lookupBy (return size) q t


main' = do ((read -> n):_) <- getArgs
           corpus <- readCorpus
           mapM print . fmap (BS.take n) $ BS.tails corpus
--           mapM print . fmap (BS.take 4) . sort . init $ BS.tails corpus
           -- print . fmap head . sort . init $ BS.tails corpus