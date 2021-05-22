# Secret sharing encryption/decryption code.
# Hadar Cohen & Alon Marzin.

# Imports
import os
import random
import numpy as np
import cv2
import hyperlpr
from PIL import Image


# The next class implements a (k,n)-thresholds visual cryptography algorithm base on Naor Shamir's algorithm
class VisualCryptographySecretSharing(object):

    def __init__(self, imagePath=""):
        self.imagePath = imagePath
        if self.imagePath == "":
            self.originalImage = cv2.imread(
                "{}/Images/test_image.png".format(os.path.dirname(os.path.abspath(__file__))))
        else:
            self.originalImage = cv2.imread(self.imagePath)
        self.originalImage = cv2.cvtColor(self.originalImage, cv2.COLOR_BGR2RGB)

        self.secretSharesImageName = ""
        self.reconstructedImageName = ""
        self.imWidth = -1
        self.imHeight = -1
        self.channel = -1
        self.recons = -1
        self.n = -1
        self.k = -1
        self.imageShares = -1
        self.secretShares = {}

    def secretSharingEncryption(self, secretSharesImageName="secretShare", k=9, n=10):
        # First validate that the (k,n)-threshold parameters are valid
        self.verifyParamsValidity(k, n)
        # Initialize original image parameters and the parameters for the secret shares creation
        self.secretSharesImageName = secretSharesImageName
        self.imWidth, self.imHeight, self.channel = self.originalImage.shape
        self.recons = n - k + 1
        self.imageShares = np.zeros([self.n, self.imWidth, self.imHeight, self.channel * 8], np.uint8)
        # Create the secret shares array that will hold all of the secret shares
        self.createSecretSharedArray()
        # Convert each secret share into a .png image
        self.createSecretSharesImages()

    # Validate that the (k,n)-threshold parameters are valid and if so assign it to the class members
    def verifyParamsValidity(self, k, n):
        if n <= 1:
            raise Exception("Invalid value for n the total number of shares\n")
        elif k < 1 or k > n:
            raise Exception("Invalid value for k the secret shares reconstruction threshold\n")
        else:
            self.n = n
            self.k = k

    # Create the secret shares and hold them in a single array
    def createSecretSharedArray(self):
        for i in range(0, self.imWidth):
            for j in range(0, self.imHeight):
                pixel = format(self.originalImage[i, j, 0], "08b") + format(self.originalImage[i, j, 1], "08b") \
                        + format(self.originalImage[i, j, 2], "08b")
                for k in range(0, self.channel * 8):
                    if pixel[k] == '1':
                        temp = np.random.permutation(self.n)[0:self.recons]
                        for r in range(0, self.recons):
                            self.imageShares[temp[r], i, j, k] = 1
                    elif pixel[k] != '0':
                        raise Exception("Binary string contains a strange number\n")

    # Split the color pixel into RGB
    def pixToRGB(self, pixel):
        vec = np.zeros([self.imWidth, self.imHeight])
        for i in range(0, self.imWidth):
            for j in range(0, self.imHeight):
                vec[i, j] = sum(val * (2 ** idx) for idx, val in enumerate(reversed(pixel[i, j, :])))
        return vec

    # Take each share and convert it back into a colour image (a final secret share)
    def createSecretSharesImages(self):
        if not os.path.isdir("./SecretShares"):
            os.mkdir("SecretShares")
        print("Starting to create secret shares images")
        for i in range(0, self.n):
            redShare = self.pixToRGB(self.imageShares[i, :, :, 0:8])
            greenShare = self.pixToRGB(self.imageShares[i, :, :, 8:16])
            blueShare = self.pixToRGB(self.imageShares[i, :, :, 16:24])

            ithShare = np.dstack((redShare, greenShare, blueShare))
            ithShare = ithShare.astype(np.uint8)
            secretShareImg = Image.fromarray(ithShare)
            secretShareImg.save("{}/SecretShares/{}{}.png".format(os.path.dirname(os.path.abspath(__file__)),
                                                                  self.secretSharesImageName, i + 1))
            self.secretShares[i] = ithShare
        print("Secret shares images created!")

    # Add documentation
    def reconstructOriginalImage(self, secretSharesImageName="secretShare", reconstructedImageName="reconstructedImage",
                                 k=9, n=10):
        # First validate that the (k,n)-threshold parameters are valid
        self.verifyParamsValidity(k, n)
        self.secretSharesImageName = secretSharesImageName
        self.reconstructedImageName = reconstructedImageName
        secretShare1 = cv2.imread(
            "{}/SecretShares/{}1.png".format(os.path.dirname(os.path.abspath(__file__)), self.secretSharesImageName))
        secretShare1 = cv2.cvtColor(secretShare1, cv2.COLOR_BGR2RGB)
        self.imWidth, self.imHeight, self.channel = secretShare1.shape

        secretSharesArray = np.zeros([self.k, self.imWidth, self.imHeight, self.channel], np.uint8)
        reconstructedImage = np.zeros([self.imWidth, self.imHeight, self.channel], np.uint8)

        print("Starting to reconstruct the original image using {} out of {} shares".format(k, n))
        indexes = list(range(1, self.n + 1))
        for i in range(0, self.k):
            shareIndex = random.choice(indexes)
            indexes.remove(shareIndex)
            randSecretShare = cv2.imread("{}/SecretShares/{}{}.png".format(os.path.dirname(os.path.abspath(__file__)),
                                                                           self.secretSharesImageName, shareIndex))
            randSecretShare = cv2.cvtColor(randSecretShare, cv2.COLOR_BGR2RGB)
            for j in range(0, self.imWidth):
                for r in range(0, self.imHeight):
                    secretSharesArray[i, j, r, 0] = randSecretShare[j, r, 0]
                    secretSharesArray[i, j, r, 1] = randSecretShare[j, r, 1]
                    secretSharesArray[i, j, r, 2] = randSecretShare[j, r, 2]

        for i in range(0, self.k):
            for j in range(0, self.imWidth):
                for r in range(0, self.imHeight):
                    reconstructedImage[j, r, 0] = reconstructedImage[j, r, 0] | secretSharesArray[i, j, r, 0]
                    reconstructedImage[j, r, 1] = reconstructedImage[j, r, 1] | secretSharesArray[i, j, r, 1]
                    reconstructedImage[j, r, 2] = reconstructedImage[j, r, 2] | secretSharesArray[i, j, r, 2]

        if not os.path.isdir("./ReconstructedImages"):
            os.mkdir("ReconstructedImages")
        reconstructedImage = reconstructedImage.astype(np.uint8)
        reconstructedImage = Image.fromarray(reconstructedImage)
        reconstructedImage.save("{}/ReconstructedImages/{}.png".format(os.path.dirname(os.path.abspath(__file__)),
                                                                       self.reconstructedImageName))


def self(args):
    pass


class LprTest(object):
    def __init__(self, imagePath="", secretSharesImageName="secretShare", reconstructedImageName="reconstructedImage",
                 k=9, n=10):
        self.originalImagePath = imagePath
        self.visualCrypto = VisualCryptographySecretSharing(imagePath)
        self.visualCrypto.verifyParamsValidity(k, n)
        self.secretSharesImageName = secretSharesImageName
        self.reconstructedImageName = reconstructedImageName
        self.n = n
        self.k = k
        if self.originalImagePath == "":
            self.originalImage = cv2.imread(
                "{}/Images/test_image.png".format(os.path.dirname(os.path.abspath(__file__))))
        else:
            self.originalImage = cv2.imread(self.originalImagePath)
        self.originalImagePlateNumbers = -1
        self.secretSharesImagesPlateNumbers = {}
        self.reconstructedImagesPlateNumbers = {}

    def lprPerformTest(self, numSharesToReconsList=[]):
        if len(numSharesToReconsList) == 0:
            numSharesToReconsList = list(range(2, self.n + 1))

        print("Starting LPR test")

        self.visualCrypto.secretSharingEncryption(self.secretSharesImageName, self.k, self.n)

        tempPlateNumberResults = hyperlpr.HyperLPR_plate_recognition(self.originalImage)
        if len(tempPlateNumberResults) > 0 and len(tempPlateNumberResults[0]) > 0:
            self.originalImagePlateNumbers = tempPlateNumberResults[0][0]
        else:
            self.originalImagePlateNumbers = 'N/A'

        secretSharesImagesList = os.listdir('{}/SecretShares/'.format(os.path.dirname(os.path.abspath(__file__))))
        for secretShareImage in secretSharesImagesList:
            currentSecretShareImage = cv2.imread(
                '{}/SecretShares/'.format(os.path.dirname(os.path.abspath(__file__))) + secretShareImage)
            tempPlateNumberResults = hyperlpr.HyperLPR_plate_recognition(currentSecretShareImage)
            if len(tempPlateNumberResults) > 0 and len(tempPlateNumberResults[0]) > 0:
                self.secretSharesImagesPlateNumbers[secretShareImage] = \
                    hyperlpr.HyperLPR_plate_recognition(currentSecretShareImage)[0][0]
            else:
                self.secretSharesImagesPlateNumbers[secretShareImage] = 'N/A'

        for numSharesToRecons in numSharesToReconsList:
            currReconstructedImageName = self.reconstructedImageName + "_{}_shares_used".format(numSharesToRecons)
            self.visualCrypto.reconstructOriginalImage(self.secretSharesImageName, currReconstructedImageName,
                                                       numSharesToRecons, self.n)
            currentReconstructedImage = cv2.imread('{}/ReconstructedImages/'.format(
                os.path.dirname(os.path.abspath(__file__))) + currReconstructedImageName + '.png')
            tempPlateNumberResults = hyperlpr.HyperLPR_plate_recognition(currentReconstructedImage)
            if len(tempPlateNumberResults) > 0 and len(tempPlateNumberResults[0]) > 0:
                self.reconstructedImagesPlateNumbers[currReconstructedImageName] = tempPlateNumberResults[0][0]
            else:
                self.reconstructedImagesPlateNumbers[currReconstructedImageName] = 'N/A'
        print("LPR test finished!")

    def printLprTestResults(self):
        print("\nLPR test results:\n")
        print("Original image licence plate number [{}]".format(self.originalImagePlateNumbers))
        originalLength = len(self.originalImagePlateNumbers)
        for key, val in self.secretSharesImagesPlateNumbers.items():
            numMatched = 0
            for i in range(0, len(val)):
                if self.originalImagePlateNumbers[i] == val[i]:
                    numMatched += 1
            print(
                "Secret Share image [{}] licence plate number [{}], number of matched numbers in plate is [{}\{}]".format(
                    key, val, numMatched, originalLength))

        for key, val in self.reconstructedImagesPlateNumbers.items():
            numMatched = 0
            for i in range(0, len(val)):
                if self.originalImagePlateNumbers[i] == val[i]:
                    numMatched += 1
            print(
                "Reconstructed image [{}] licence plate number [{}], number of matched numbers in plate is [{}\{}]".format(
                    key, val, numMatched, originalLength))
            if numMatched == originalLength:
                print("All the numbers in the reconstructed image match with the original image!")
                break


# Main
if __name__ == '__main__':
    lprTest = LprTest(k=5, n=10)
    lprTest.lprPerformTest()
    lprTest.printLprTestResults()
