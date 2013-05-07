{-# LANGUAGE TupleSections #-}

import System.IO.MMap

import System.Environment
-- import Control.Monad

import qualified Data.ByteString as BS
-- import qualified Data.ByteString.Char8 as BS8
-- import Data.Trie as T
import Data.Trie.Convenience as T

-- import System.Environment


readCorpus :: FilePath -> IO BS.ByteString
readCorpus corpusName = mmapFileByteString corpusName Nothing

main :: IO ()
main = do
  (corpusName:_) <- getArgs
  print . T.fromListWith (+) . fmap (,(1 :: Int)) . fmap (BS.take 1) . BS.tails =<< readCorpus corpusName

