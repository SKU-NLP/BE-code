package com.app.NLP.global.exception;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;

@Getter
@RequiredArgsConstructor
public enum Success {

    OK(HttpStatus.OK, "성공"),

    /***
     * 201 CREATED
     */
    CREATED(HttpStatus.CREATED, "생성 성공");

    private final HttpStatus httpStatus;
    private final String message;

    public int getHttpStatusCode() {
        return httpStatus.value();
    }
}
