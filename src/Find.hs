{-# LANGUAGE OverloadedStrings,NoMonomorphismRestriction #-}

module Find where

import Data.List (isInfixOf)
import Control.Monad
import Control.Monad.IO.Class (liftIO)
import Data.ByteString (ByteString)
import qualified Data.ByteString as BS
-- for OverloadedStrings
import Data.ByteString.Char8 ()
import qualified Data.ByteString.Char8 as BS8
import Data.ByteString.Char8 (putStrLn)
import Data.Enumerator hiding (map, filter, filterM)
import qualified Data.Enumerator.Binary as EB
import qualified Data.Enumerator.List as EL
import Data.Maybe
import Control.Applicative
import System.Directory
import System.FilePath

l = 2

getValidContents :: FilePath -> IO [String]
getValidContents path = 
    filter (`notElem` [".", "..", ".git", ".svn"])
    <$> getDirectoryContents path

isSearchableDir :: FilePath -> IO Bool
isSearchableDir dir =
    (&&) <$> doesDirectoryExist dir
         <*> (searchable <$> getPermissions dir)

consumer2 :: Iteratee ByteString IO ()
consumer2 = do
    mw <- EB.head
    case mw of
        Nothing -> return ()
        Just w  -> do
            liftIO . putStr $ "YYY "
            liftIO . BS8.putStrLn . BS.singleton $ w

consumer :: Iteratee BS.ByteString IO ()
consumer = do
    mw <- EB.head
    case mw of
        Nothing -> return ()
        Just w  -> do
            -- liftIO . putStr $ "XXX "
            liftIO . BS8.putStr . BS.singleton $ w
            consumer

fileFeeder = EB.enumFile "/etc/mtab"

listFeeder :: Monad m => Enumerator BS.ByteString m ()
listFeeder = enumList 1 ["hello world"," bla\n\n"]

-- main = findEnum "/var/log/" "e" -- run_ $ fileFeeder <==< listFeeder $$ EB.isolate 1 =$ (consumer2 >> consumer)

tree = enumDir

grepE :: String -> Enumeratee String String IO b
grepE pattern = EL.filter (pattern `isInfixOf`)

printI :: Iteratee String IO ()
printI = do
    mx <- EL.head 
    case mx of
        Nothing   -> return ()
        Just file -> do
            liftIO . Prelude.putStrLn $ file
            printI
            
enumDir :: FilePath -> Enumerator FilePath IO b
enumDir dir = list
  where
    list (Continue k) = do
        (files, dirs) <- liftIO getFilesDirs
        (case dirs of
              [] -> id
              (_:_) -> (walk dirs ==<<)) $
            k (Chunks files)
    list step = returnI step
    walk dirs = foldr1 (<==<) $ map enumDir dirs
    getFilesDirs = do
        cnts <- map (dir </>) <$> getValidContents dir
        (,) <$> filterM doesFileExist cnts
            <*> filterM isSearchableDir cnts

findEnum :: FilePath -> String -> IO ()
findEnum dir pattern = run_ $ enumDir dir -- $$ printI
                           $$ grepE pattern
                           =$ printI